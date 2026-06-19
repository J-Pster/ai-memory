//! Integration tests for `POST /admin/rehome-by-kind`.
//!
//! Same pattern as `admin_rename.rs`: build a real [`AdminState`] over a
//! tmpdir-backed store + wiki, drive the router with
//! `tower::ServiceExt::oneshot`, and verify side effects through the reader.

use ai_memory_core::{PagePath, Tier, WorkspaceId};
use ai_memory_mcp::{AdminState, admin_router};
use ai_memory_store::{DecayParams, Store};
use ai_memory_wiki::{Wiki, WritePageRequest};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tempfile::TempDir;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn make_state(tmp: &TempDir) -> (AdminState, Store) {
    let store = Store::open(tmp.path()).unwrap();
    let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
    let db_path = store.db_path().to_path_buf();
    let state = AdminState {
        writer: store.writer.clone(),
        reader: store.reader.clone(),
        wiki,
        llm: None,
        auto_improve_require_approval: false,
        embedder: None,
        provider_health: ai_memory_llm::ProviderHealth::default(),
        decay_params: DecayParams::default(),
        data_dir: tmp.path().to_path_buf(),
        db_path,
        bind: "127.0.0.1:0".to_string(),
        home_dir: None,
        bootstrap_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        token_pepper: None,
        active_project: ai_memory_core::ActiveProject::new(),
        on_project_moved: None,
    };
    (state, store)
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn post(state: AdminState, uri: &str, body: serde_json::Value) -> axum::response::Response {
    let router = admin_router(state);
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    router.oneshot(req).await.unwrap()
}

/// Write one page at `path` with the given `kind` and body.
async fn seed(store: &Store, wiki: &Wiki, project: &str, path: &str, kind: &str, body: &str) {
    let ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, project, None)
        .await
        .unwrap();
    wiki.write_page(WritePageRequest {
        workspace_id: ws,
        project_id: proj,
        path: PagePath::new(path.to_string()).unwrap(),
        frontmatter: json!({ "title": path, "kind": kind }),
        body: body.to_string(),
        tier: Tier::Semantic,
        pinned: false,
        title: Some(path.to_string()),
        admission_ctx: None,
        author_id: None,
        actor: ai_memory_core::ActorContext::anonymous(),
    })
    .await
    .unwrap();
}

async fn ws_proj(store: &Store, project: &str) -> (WorkspaceId, ai_memory_core::ProjectId) {
    let ws = store
        .reader
        .find_workspace("default".to_string())
        .await
        .unwrap()
        .expect("workspace");
    let proj = store
        .reader
        .find_project(ws, project.to_string())
        .await
        .unwrap()
        .expect("project");
    (ws, proj)
}

async fn paths(store: &Store, project: &str) -> Vec<String> {
    store
        .reader
        .list_pages("default", project)
        .await
        .unwrap()
        .into_iter()
        .map(|p| p.path)
        .collect()
}

async fn body_of(store: &Store, project: &str, path: &str) -> Option<String> {
    let (ws, proj) = ws_proj(store, project).await;
    store
        .reader
        .page_body_by_ids(ws, proj, path)
        .await
        .unwrap()
        .map(|s| s.body)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Happy path: a decision + a fact under `imported/` move to `decisions/`
/// and `concepts/`; the folder-qualified wikilink is rewritten while the
/// bare wikilink is left untouched; old paths are deleted.
#[tokio::test]
async fn rehome_moves_by_kind_and_rewrites_links() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;

    seed(
        &store,
        &state.wiki,
        "p",
        "imported/foo.md",
        "decision",
        "# Foo\n\nSee [[imported/bar.md|Bar]] and bare [[bar]].\n",
    )
    .await;
    seed(
        &store,
        &state.wiki,
        "p",
        "imported/bar.md",
        "fact",
        "# Bar\n",
    )
    .await;

    let resp = post(
        state,
        "/admin/rehome-by-kind",
        json!({ "workspace": "default", "project": "p", "dry_run": false }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let b = body_json(resp).await;
    assert_eq!(b["pages_moved"].as_u64(), Some(2), "{b}");
    assert_eq!(b["links_rewritten"].as_u64(), Some(1), "{b}");

    let now = paths(&store, "p").await;
    assert!(now.contains(&"decisions/foo.md".to_string()), "{now:?}");
    assert!(now.contains(&"concepts/bar.md".to_string()), "{now:?}");
    assert!(!now.contains(&"imported/foo.md".to_string()), "{now:?}");
    assert!(!now.contains(&"imported/bar.md".to_string()), "{now:?}");

    let foo = body_of(&store, "p", "decisions/foo.md").await.unwrap();
    assert!(foo.contains("[[concepts/bar.md|Bar]]"), "rewritten: {foo}");
    assert!(foo.contains("[[bar]]"), "bare link preserved: {foo}");
}

/// `dry_run` reports the plan but writes nothing.
#[tokio::test]
async fn rehome_dry_run_writes_nothing() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;

    seed(
        &store,
        &state.wiki,
        "p",
        "imported/foo.md",
        "gotcha",
        "# Foo\n",
    )
    .await;

    let resp = post(
        state,
        "/admin/rehome-by-kind",
        json!({ "workspace": "default", "project": "p", "dry_run": true }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let b = body_json(resp).await;
    assert_eq!(b["dry_run"].as_bool(), Some(true));
    assert_eq!(b["pages_moved"].as_u64(), Some(1), "plan size: {b}");

    let now = paths(&store, "p").await;
    assert!(
        now.contains(&"imported/foo.md".to_string()),
        "still at old path: {now:?}"
    );
    assert!(!now.contains(&"gotchas/foo.md".to_string()), "{now:?}");
}

/// Running rehome twice is a no-op the second time (idempotent).
#[tokio::test]
async fn rehome_is_idempotent() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;

    seed(
        &store,
        &state.wiki,
        "p",
        "imported/foo.md",
        "rule",
        "# Foo\n",
    )
    .await;

    let first = post(
        state,
        "/admin/rehome-by-kind",
        json!({ "workspace": "default", "project": "p", "dry_run": false }),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(body_json(first).await["pages_moved"].as_u64(), Some(1));

    let (state2, _store2) = rebuild_state(&store, &tmp).await;
    let second = post(
        state2,
        "/admin/rehome-by-kind",
        json!({ "workspace": "default", "project": "p", "dry_run": false }),
    )
    .await;
    assert_eq!(second.status(), StatusCode::OK);
    let b = body_json(second).await;
    assert_eq!(
        b["pages_moved"].as_u64(),
        Some(0),
        "second run is a no-op: {b}"
    );

    let now = paths(&store, "p").await;
    assert!(now.contains(&"_rules/foo.md".to_string()), "{now:?}");
}

/// Two pages with the same slug onto one folder are both skipped, never
/// clobbered.
#[tokio::test]
async fn rehome_skips_colliding_targets() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;

    seed(&store, &state.wiki, "p", "imported/x.md", "fact", "# A\n").await;
    seed(&store, &state.wiki, "p", "omc/x.md", "fact", "# B\n").await;

    let resp = post(
        state,
        "/admin/rehome-by-kind",
        json!({ "workspace": "default", "project": "p", "dry_run": false }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let b = body_json(resp).await;
    assert_eq!(b["pages_moved"].as_u64(), Some(0), "{b}");
    assert_eq!(b["skipped"].as_array().map(|a| a.len()), Some(2), "{b}");

    let now = paths(&store, "p").await;
    assert!(now.contains(&"imported/x.md".to_string()), "{now:?}");
    assert!(now.contains(&"omc/x.md".to_string()), "{now:?}");
}

/// A missing workspace returns 404 (no auto-create).
#[tokio::test]
async fn rehome_missing_workspace_returns_404() {
    let tmp = TempDir::new().unwrap();
    let (state, _store) = make_state(&tmp).await;

    let resp = post(
        state,
        "/admin/rehome-by-kind",
        json!({ "workspace": "ghost", "project": "p", "dry_run": false }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Build a fresh `AdminState` over the same store (axum's `oneshot`
/// consumes the router, so a second call needs a second state that shares
/// the single writer + reader).
async fn rebuild_state(store: &Store, tmp: &TempDir) -> (AdminState, ()) {
    let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
    let state = AdminState {
        writer: store.writer.clone(),
        reader: store.reader.clone(),
        wiki,
        llm: None,
        auto_improve_require_approval: false,
        embedder: None,
        provider_health: ai_memory_llm::ProviderHealth::default(),
        decay_params: DecayParams::default(),
        data_dir: tmp.path().to_path_buf(),
        db_path: store.db_path().to_path_buf(),
        bind: "127.0.0.1:0".to_string(),
        home_dir: None,
        bootstrap_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        token_pepper: None,
        active_project: ai_memory_core::ActiveProject::new(),
        on_project_moved: None,
    };
    (state, ())
}
