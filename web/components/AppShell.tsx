"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";

import Sidebar from "@/components/Sidebar";
import { useAuth } from "@/context/AuthContext";

const AUTH_FREE_ROUTES = new Set<string>(["/login"]);

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated, isLoading } = useAuth();

  const isAuthFreeRoute = AUTH_FREE_ROUTES.has(pathname);

  useEffect(() => {
    if (isLoading) {
      return;
    }

    if (!isAuthenticated && !isAuthFreeRoute) {
      router.replace("/login");
      return;
    }

    if (isAuthenticated && pathname === "/login") {
      router.replace("/");
    }
  }, [isAuthFreeRoute, isAuthenticated, isLoading, pathname, router]);

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-background text-foreground">
        <div className="text-sm text-slate-500 dark:text-slate-400">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated && isAuthFreeRoute) {
    return <main className="min-h-screen">{children}</main>;
  }

  if (isAuthenticated && isAuthFreeRoute) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-background text-foreground">
        <div className="text-sm text-slate-500 dark:text-slate-400">Redirecting...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-background text-foreground">
        <div className="text-sm text-slate-500 dark:text-slate-400">Redirecting to login...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden transition-colors duration-200 bg-transparent">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-transparent">{children}</main>
    </div>
  );
}
