//! Authorization middleware for the HTTP server.
//!
//! When `[auth].bearer_token` (or the `AI_MEMORY_AUTH_TOKEN` env var)
//! is set, every request to `/mcp`, `/hook`, `/handoff`, and `/web/*`
//! must present the token via one of three transports:
//!
//! - **Bearer header** (any method): MCP clients + hooks. Required
//!   on all non-GET methods.
//! - **Basic auth** (GET only): browsers — username ignored, token
//!   in the password field. Triggers the native credential dialog
//!   via the `WWW-Authenticate: Basic` challenge in 401 responses.
//! - **Session cookie** (GET only): set automatically after a
//!   successful Basic auth so the browser doesn't re-prompt every
//!   session.
//!
//! When the token is *unset*, the middleware is a no-op — preserving
//! the zero-config local-development experience and keeping the
//! existing e2e + unit tests working.
//!
//! Comparison uses [`subtle::ConstantTimeEq`] so an attacker on the
//! same LAN cannot use response-time leaks to recover the token byte
//! by byte. The constant-time guarantee depends on both sides being
//! the same length; `subtle` returns a constant-cost `Choice::from(0)`
//! when lengths differ, which is the right thing here.
//!
//! Wire shape matches the MCP authorization spec
//! (modelcontextprotocol.io/specification/.../basic/authorization):
//! 401 responses include `WWW-Authenticate: Bearer …` so MCP clients
//! detect missing/expired credentials. GET 401s ALSO include `Basic
//! …` so browsers dialog-prompt automatically.
//!
//! ## Why not OAuth
//!
//! The MCP spec mandates full OAuth 2.1 for HTTP-authenticated
//! servers. That's overkill for a single-user homelab and would
//! force every MCP client config to deal with authorization-server
//! discovery + PKCE + token refresh. A static bearer token is
//! wire-compatible with the spec's `Authorization: Bearer …` shape
//! (clients send the header the same way; they just don't run the
//! OAuth dance to obtain the token). Every supported client
//! (Claude Code, Codex, OpenCode, Cursor, Claude Desktop via
//! `mcp-remote`, Gemini CLI, OpenClaw) accepts a static
//! `Authorization` header in its config.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{Method, Request, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;
use tracing::debug;

/// Cookie name used for browser session persistence after a
/// successful Basic auth handshake.
const AUTH_COOKIE: &str = "ai_memory_auth";
/// Realm advertised in `WWW-Authenticate` challenges. Shows up in
/// the browser's credential prompt as "Server says: <realm>".
const AUTH_REALM: &str = "ai-memory";

/// Shared auth state. Cheap to clone — just an `Arc` wrapping the
/// optional configured token.
#[derive(Clone, Debug)]
pub struct AuthState {
    expected: Option<String>,
}

impl AuthState {
    /// Build state from the (optional) configured token. `None` means
    /// "auth disabled, accept everything".
    #[must_use]
    pub fn new(expected: Option<String>) -> Self {
        Self { expected }
    }

    /// True when a token is configured (i.e. the middleware is doing
    /// anything). Useful for the startup log line so the operator
    /// sees whether their server is open or closed.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.expected.is_some()
    }
}

/// axum middleware closure. Wire with
/// `axum::middleware::from_fn_with_state(state, require_bearer)`.
///
/// Token sources, in priority order:
/// 1. `Authorization: Bearer <token>` header. Works for any method.
///    This is what MCP + hook clients send.
/// 2. **GET only:** `Authorization: Basic <base64(user:token)>`.
///    Username is ignored; the password portion is the token.
///    Browsers send this automatically after the native credential
///    prompt fires on a 401 + `WWW-Authenticate: Basic`. On success
///    we also set the `ai_memory_auth` cookie so subsequent visits
///    (including from a fresh browser session) skip the prompt.
/// 3. **GET only:** `ai_memory_auth` cookie set by the Basic handshake.
///
/// POST / PUT / DELETE / etc. require the Bearer header. Cookie and
/// Basic auth are GET-only, which confines cookie-CSRF to read-only
/// pages — `/mcp` + `/hook` are POST-only and stay header-gated.
///
/// On 401 for GET requests the response includes both `Basic` and
/// `Bearer` challenges in `WWW-Authenticate`. Browsers honour the
/// `Basic` challenge (native dialog); MCP clients honour the `Bearer`
/// challenge.
pub async fn require_bearer(
    State(state): State<Arc<AuthState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let Some(expected) = state.expected.as_deref() else {
        return next.run(req).await;
    };

    let is_get = req.method() == Method::GET;
    let from_bearer = extract_bearer_header(&req);
    let from_basic = if is_get {
        extract_basic_header(&req)
    } else {
        None
    };
    let from_cookie = if is_get { extract_cookie(&req) } else { None };

    let provided = from_bearer
        .as_deref()
        .or(from_basic.as_deref())
        .or(from_cookie.as_deref())
        .unwrap_or("");

    if !bool::from(provided.as_bytes().ct_eq(expected.as_bytes())) {
        debug!("auth rejected: invalid or missing token");
        return unauthorized(is_get);
    }

    // First successful Basic-auth hit (no cookie yet) → also stamp the
    // cookie so the user doesn't get the dialog again next browser
    // session. Subsequent navigations ride the cookie alone.
    if from_basic.is_some() && from_cookie.is_none() {
        let mut resp = next.run(req).await;
        if let Ok(cookie) = build_session_cookie(provided).parse() {
            resp.headers_mut().insert(header::SET_COOKIE, cookie);
        }
        return resp;
    }

    next.run(req).await
}

fn extract_bearer_header(req: &Request<axum::body::Body>) -> Option<String> {
    let h = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    // Accept both "Bearer xxx" and "bearer xxx" (case-insensitive
    // scheme per RFC 7235 §2.1).
    let (scheme, value) = h.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("Bearer") {
        Some(value.trim_start().to_string())
    } else {
        None
    }
}

fn extract_basic_header(req: &Request<axum::body::Body>) -> Option<String> {
    use base64::Engine;
    let h = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, value) = h.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("Basic") {
        return None;
    }
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(value.trim_start())
        .ok()?;
    let s = std::str::from_utf8(&decoded).ok()?;
    // Standard form: `user:password`. We ignore the username (the
    // browser dialog always asks for one but we don't have multi-user
    // accounts — only the password = bearer token matters).
    let (_user, pass) = s.split_once(':')?;
    Some(pass.to_string())
}

fn extract_cookie(req: &Request<axum::body::Body>) -> Option<String> {
    let h = req.headers().get(header::COOKIE)?.to_str().ok()?;
    for pair in h.split(';') {
        let pair = pair.trim();
        if let Some(val) = pair.strip_prefix(&format!("{AUTH_COOKIE}=")) {
            return Some(val.to_string());
        }
    }
    None
}

fn build_session_cookie(token: &str) -> String {
    // 30-day Max-Age — long enough that re-entering the credential
    // every month is rare. HttpOnly hides it from any inline JS;
    // SameSite=Lax keeps cross-site POSTs from riding it.
    // No Secure attribute: homelab deployments are often plain HTTP
    // on a LAN. A TLS-terminating reverse proxy is the right place to
    // add Secure if the service is exposed publicly.
    format!("{AUTH_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000")
}

fn unauthorized(include_basic_challenge: bool) -> Response {
    let mut resp = (StatusCode::UNAUTHORIZED, "auth required\n").into_response();
    // Order of challenges matters: browsers parse the first challenge
    // they understand and show the dialog for it. Putting `Basic`
    // first ensures GET-from-browser triggers the native prompt; MCP
    // clients (which speak only Bearer) ignore the Basic and read
    // their challenge from the second value.
    //
    // Non-GET 401s skip the Basic challenge — sending it on a POST
    // would invite the browser to dialog-prompt for an endpoint
    // it can't authenticate this way anyway.
    let value = if include_basic_challenge {
        format!(
            "Basic realm=\"{AUTH_REALM}\", \
             Bearer realm=\"{AUTH_REALM}\", error=\"invalid_token\""
        )
    } else {
        format!("Bearer realm=\"{AUTH_REALM}\", error=\"invalid_token\"")
    };
    resp.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        value.parse().expect("static header value is valid"),
    );
    resp
}

/// Generate a fresh random bearer token, hex-encoded.
///
/// `bytes` is the entropy budget; 32 bytes (256 bits) is plenty for
/// any conceivable threat model.
///
/// # Errors
/// Propagates failures from the OS RNG.
pub fn generate_token_hex(bytes: usize) -> Result<String, getrandom::Error> {
    let mut buf = vec![0u8; bytes];
    getrandom::fill(&mut buf)?;
    Ok(hex_encode(&buf))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use tower::ServiceExt;

    fn router_with_auth(token: Option<&str>) -> Router {
        let state = Arc::new(AuthState::new(token.map(str::to_string)));
        Router::new()
            .route("/probe", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(state, require_bearer))
    }

    #[tokio::test]
    async fn no_token_configured_passes_anything_through() {
        let r = router_with_auth(None);
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn missing_header_returns_401_with_www_authenticate() {
        let r = router_with_auth(Some("secret"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let www = resp.headers().get(header::WWW_AUTHENTICATE).unwrap();
        let www = www.to_str().unwrap();
        // GET 401 advertises BOTH challenges so browsers (Basic) and
        // MCP clients (Bearer) each see what they understand.
        assert!(www.contains("Bearer"));
        assert!(www.contains("Basic"));
    }

    #[tokio::test]
    async fn wrong_token_returns_401() {
        let r = router_with_auth(Some("the-right-one"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Authorization", "Bearer the-wrong-one")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn right_token_returns_200() {
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Authorization", "Bearer right-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn lowercase_scheme_is_accepted() {
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Authorization", "bearer right-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_scheme_is_rejected() {
        // `Digest`, `OAuth`, etc. are not handled.
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Authorization", "Digest username=foo,response=bar")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn cookie_with_right_token_passes_get() {
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Cookie", "ai_memory_auth=right-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cookie_with_wrong_token_fails() {
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Cookie", "ai_memory_auth=wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn cookie_ignored_on_post() {
        // POST routes must use Bearer header; cookie auth is GET-only
        // to keep the CSRF surface confined to read paths.
        let state = Arc::new(AuthState::new(Some("right-token".to_string())));
        let r = Router::new()
            .route("/probe", axum::routing::post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(state, require_bearer));
        let resp = r
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/probe")
                    .header("Cookie", "ai_memory_auth=right-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// Helper: build a Basic-auth header value (any username, token as password).
    fn basic_auth(token: &str) -> String {
        use base64::Engine;
        let creds = format!("any:{token}");
        format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode(creds)
        )
    }

    #[tokio::test]
    async fn basic_auth_with_right_token_passes_get_and_sets_cookie() {
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Authorization", basic_auth("right-token"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // First successful Basic hit also stamps the cookie so the
        // browser doesn't dialog-prompt every session.
        let cookie = resp
            .headers()
            .get(header::SET_COOKIE)
            .expect("set-cookie on first Basic-auth success")
            .to_str()
            .unwrap();
        assert!(cookie.contains("ai_memory_auth=right-token"));
        assert!(cookie.contains("HttpOnly"));
        assert!(cookie.contains("SameSite=Lax"));
        assert!(cookie.contains("Path=/"));
    }

    #[tokio::test]
    async fn basic_auth_with_wrong_password_returns_401() {
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Authorization", basic_auth("wrong-token"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn basic_auth_ignored_on_post() {
        // POST routes must use Bearer header; Basic auth is GET-only.
        let state = Arc::new(AuthState::new(Some("right-token".to_string())));
        let r = Router::new()
            .route("/probe", axum::routing::post(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(state, require_bearer));
        let resp = r
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/probe")
                    .header("Authorization", basic_auth("right-token"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // POST 401 must NOT advertise Basic — browsers would dialog
        // for a route they can't authenticate this way.
        let www = resp.headers().get(header::WWW_AUTHENTICATE).unwrap();
        let www = www.to_str().unwrap();
        assert!(www.contains("Bearer"));
        assert!(!www.contains("Basic"));
    }

    #[tokio::test]
    async fn cookie_request_does_not_re_set_cookie() {
        // Already-authed-by-cookie requests don't need a Set-Cookie
        // refresh; that's a waste of bandwidth on every navigation.
        let r = router_with_auth(Some("right-token"));
        let resp = r
            .oneshot(
                Request::builder()
                    .uri("/probe")
                    .header("Cookie", "ai_memory_auth=right-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get(header::SET_COOKIE).is_none());
    }

    #[test]
    fn generated_token_is_hex_and_correct_length() {
        let t = generate_token_hex(32).unwrap();
        assert_eq!(t.len(), 64);
        assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
        // Distinct calls produce distinct tokens (modulo OS RNG bugs).
        let t2 = generate_token_hex(32).unwrap();
        assert_ne!(t, t2);
    }
}
