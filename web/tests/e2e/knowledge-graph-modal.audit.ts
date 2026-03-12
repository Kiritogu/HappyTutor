import { expect, test } from "@playwright/test";

const BASE_URL =
  process.env.WEB_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:3000";

test.describe("Knowledge :: Graph modal entry", () => {
  test("lightrag kb shows graph button and opens modal", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("deeptutor-auth-access-token", "fake-access-token");
      localStorage.setItem(
        "deeptutor-auth-refresh-token",
        "fake-refresh-token",
      );
      localStorage.setItem(
        "deeptutor-auth-user",
        JSON.stringify({
          id: "u1",
          email: "user@example.com",
          is_email_verified: true,
          status: "active",
        }),
      );
    });

    await page.route("**/api/v1/auth/me", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          user: {
            id: "u1",
            email: "user@example.com",
            is_email_verified: true,
            status: "active",
          },
        }),
      });
    });

    await page.route("**/api/v1/settings/sidebar", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          description: "test",
          nav_order: {
            start: ["/", "/history", "/knowledge", "/notebook"],
            learnResearch: ["/question", "/guide", "/research"],
          },
        }),
      });
    });

    await page.route("**/api/v1/knowledge/health", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          storage_backend: "postgres",
          is_degraded: false,
          rag_storage_exists: true,
        }),
      });
    });

    await page.route("**/api/v1/knowledge/rag-providers", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          providers: [
            { id: "llamaindex", name: "LlamaIndex", description: "" },
            { id: "lightrag", name: "LightRAG", description: "" },
            { id: "raganything", name: "RAG-Anything", description: "" },
          ],
        }),
      });
    });

    await page.route("**/api/v1/knowledge/list", async (route) => {
      await route.fulfill({
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify([
          {
            name: "kb_graph",
            is_default: true,
            statistics: {
              raw_documents: 2,
              images: 1,
              content_lists: 0,
              rag_initialized: true,
              rag_provider: "lightrag",
              status: "ready",
              rag: { chunks: 10, entities: 4, relations: 3 },
            },
          },
        ]),
      });
    });

    await page.goto(`${BASE_URL}/knowledge`);

    const graphButton = page
      .locator(
        'button[title="View Knowledge Graph"], button[title="查看知识图谱"]',
      )
      .first();

    await expect(graphButton).toBeVisible();
    await graphButton.click({ force: true });

    await expect(
      page
        .locator(
          'input[placeholder="Search graph anchor..."], input[placeholder="搜索图谱锚点..."]',
        )
        .first(),
    ).toBeVisible();
  });
});
