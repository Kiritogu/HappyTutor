import type { Metadata } from "next";
import "./globals.css";
import ThemeScript from "@/components/ThemeScript";
import LayoutWrapper from "@/components/LayoutWrapper";
import AppShell from "@/components/AppShell";
import { AuthProvider } from "@/context/AuthContext";
import { GlobalProvider } from "@/context/GlobalContext";
import { I18nClientBridge } from "@/i18n/I18nClientBridge";

export const metadata: Metadata = {
  title: "DeepTutor Platform",
  description: "Multi-Agent Teaching & Research Copilot",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body className="font-sans">
        <AuthProvider>
          <GlobalProvider>
            <I18nClientBridge>
              <LayoutWrapper>
                <AppShell>{children}</AppShell>
              </LayoutWrapper>
            </I18nClientBridge>
          </GlobalProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
