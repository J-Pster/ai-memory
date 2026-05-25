//! Smoke integration tests for the read-only web UI.
//!
//! Spins up a `Store` + `Wiki` in a tempdir, seeds two pages, builds
//! the router, and exercises each route via `tower::ServiceExt::oneshot`.

use ai_memory_core::{NewPage, PagePath, Tier};
use ai_memory_store::Store;
use ai_memory_web::router;
use ai_memory_wiki::{Wiki, WritePageRequest};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tempfile::TempDir;
use tower::ServiceExt;

async fn setup() -> (TempDir, Store, Wiki) {
    let tmp = TempDir::new().unwrap();
    let store = Store::open(tmp.path()).unwrap();
    let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
    (tmp, store, wiki)
}

fn new_page(
    ws: ai_memory_core::WorkspaceId,
    proj: ai_memory_core::ProjectId,
    path: &str,
    title: &str,
    body: &str,
) -> NewPage {
    NewPage {
        workspace_id: ws,
        project_id: proj,
        path: PagePath::new(path).unwrap(),
        title: title.to_owned(),
        body: body.to_owned(),
        tier: Tier::Semantic,
        frontmatter_json: serde_json::json!({"kind": "fact"}),
        pinned: false,
        links: Vec::new(),
    }
}

fn wiki_req(
    ws: ai_memory_core::WorkspaceId,
    proj: ai_memory_core::ProjectId,
    path: &str,
    body: &str,
) -> WritePageRequest {
    WritePageRequest {
        workspace_id: ws,
        project_id: proj,
        path: PagePath::new(path).unwrap(),
        frontmatter: serde_json::json!({"kind": "fact"}),
        body: body.to_owned(),
        tier: Tier::Semantic,
        pinned: false,
        title: None,
    }
}

#[tokio::test]
async fn smoke_index_returns_200() {
    let (_tmp, store, wiki) = setup().await;
    let ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch", None)
        .await
        .unwrap();
    store
        .writer
        .upsert_page(new_page(ws, proj, "foo.md", "Foo Page", "Hello world"))
        .await
        .unwrap();

    let app = router(store.reader.clone(), wiki.clone());
    let req = Request::builder().uri("/").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("scratch"),
        "expected project name in index response"
    );
}

#[tokio::test]
async fn smoke_project_page_returns_200() {
    let (_tmp, store, wiki) = setup().await;
    let ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch", None)
        .await
        .unwrap();
    store
        .writer
        .upsert_page(new_page(
            ws,
            proj,
            "notes/bar.md",
            "Bar Note",
            "A note about bar",
        ))
        .await
        .unwrap();

    let app = router(store.reader.clone(), wiki.clone());
    let req = Request::builder()
        .uri("/w/default/scratch")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("Bar Note"),
        "expected page title in project response"
    );
}

#[tokio::test]
async fn smoke_page_view_returns_200() {
    let (_tmp, store, wiki) = setup().await;
    let ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch", None)
        .await
        .unwrap();
    // Use wiki.write_page so the file is written to disk (needed for read_page).
    wiki.write_page(wiki_req(ws, proj, "foo.md", "# Foo\n\nHello world"))
        .await
        .unwrap();

    let app = router(store.reader.clone(), wiki.clone());
    let req = Request::builder()
        .uri("/w/default/scratch/p/foo.md")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    // The title is derived from the H1 heading.
    assert!(text.contains("Foo"), "expected page title");
    assert!(text.contains("Hello world"), "expected rendered body");
}

#[tokio::test]
async fn smoke_search_returns_200() {
    let (_tmp, store, wiki) = setup().await;
    let ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch", None)
        .await
        .unwrap();
    store
        .writer
        .upsert_page(new_page(
            ws,
            proj,
            "foo.md",
            "Searchable Page",
            "unique_term_xyz_abc",
        ))
        .await
        .unwrap();

    let app = router(store.reader.clone(), wiki.clone());
    let req = Request::builder()
        .uri("/search?q=unique_term_xyz_abc")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("unique_term_xyz_abc"),
        "expected search term in results"
    );
}

#[tokio::test]
async fn web_links_percent_encode_route_segments() {
    let (_tmp, store, wiki) = setup().await;
    let ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch #1", None)
        .await
        .unwrap();
    store
        .writer
        .upsert_page(new_page(
            ws,
            proj,
            "notes/a b%25.md",
            "Encoded Link",
            "route encoding check",
        ))
        .await
        .unwrap();

    let app = router(store.reader.clone(), wiki.clone());
    let req = Request::builder()
        .uri("/w/default/scratch%20%231")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.contains("/web/w/default/scratch%20%231/p/notes/a%20b%2525.md"),
        "expected encoded href in project response: {text}"
    );
}

#[tokio::test]
async fn smoke_page_not_found_returns_404() {
    let (_tmp, store, wiki) = setup().await;
    let _ws = store
        .writer
        .get_or_create_workspace("default")
        .await
        .unwrap();

    let app = router(store.reader.clone(), wiki.clone());
    let req = Request::builder()
        .uri("/w/default/scratch/p/does-not-exist.md")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
