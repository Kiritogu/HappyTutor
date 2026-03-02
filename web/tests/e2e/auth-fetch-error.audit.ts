import { expect, test } from "@playwright/test";

const BASE_URL =
  process.env.WEB_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:3000";

test.describe("Auth :: Network error resilience", () => {
  test("question page should not force logout when non-auth API request fails", async ({
    page,
  }) => {
    await page.addInitScript(() => {
      localStorage.setItem("deeptutor-auth-access-token", "fake-access-token");
      localStorage.setItem("deeptutor-auth-refresh-token", "fake-refresh-token");
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

    await page.route("**/api/v1/knowledge/list", async (route) => {
      await route.abort("failed");
    });

    await page.goto(`${BASE_URL}/question`);
    await page.waitForTimeout(1000);

    await expect(page).toHaveURL(/\/question$/);
    await expect(page.locator("main")).toBeVisible();
  });
});
