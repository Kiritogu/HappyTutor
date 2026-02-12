"use client";

import { FormEvent, useMemo, useState } from "react";
import { useAuth } from "@/context/AuthContext";

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

type Mode = "login" | "register";

export default function LoginPage() {
  const { login, register } = useAuth();

  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const title = useMemo(() => (mode === "login" ? "登录 DeepTutor" : "注册 DeepTutor 账号"), [mode]);

  const submitLabel = mode === "login" ? "登录" : "注册并登录";

  const validate = (): string | null => {
    if (!email.trim()) {
      return "请输入邮箱";
    }
    if (!EMAIL_REGEX.test(email.trim())) {
      return "邮箱格式不正确";
    }
    if (!password) {
      return "请输入密码";
    }
    if (password.length < 8) {
      return "密码至少 8 位";
    }
    return null;
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSubmitting(true);
    try {
      if (mode === "login") {
        await login(email.trim(), password);
      } else {
        await register(email.trim(), password);
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "请求失败，请稍后重试");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      <div className="w-full max-w-md rounded-2xl border border-slate-200/80 dark:border-slate-700/70 bg-white/90 dark:bg-slate-900/80 backdrop-blur p-6 shadow-xl">
        <div className="mb-6">
          <div className="text-xs uppercase tracking-widest text-emerald-600 dark:text-emerald-400 font-semibold mb-2">
            Authentication
          </div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">{title}</h1>
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
            使用邮箱和密码登录，所有数据将按账号严格隔离。
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2 p-1 rounded-xl bg-slate-100 dark:bg-slate-800 mb-5">
          <button
            type="button"
            onClick={() => setMode("login")}
            className={`rounded-lg py-2 text-sm font-medium transition ${
              mode === "login"
                ? "bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 shadow"
                : "text-slate-500 dark:text-slate-300"
            }`}
          >
            登录
          </button>
          <button
            type="button"
            onClick={() => setMode("register")}
            className={`rounded-lg py-2 text-sm font-medium transition ${
              mode === "register"
                ? "bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 shadow"
                : "text-slate-500 dark:text-slate-300"
            }`}
          >
            注册
          </button>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">
              邮箱
            </label>
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5 text-sm text-slate-900 dark:text-slate-100 outline-none focus:ring-2 focus:ring-emerald-500/30"
              placeholder="you@example.com"
              disabled={isSubmitting}
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-2">
              密码
            </label>
            <input
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5 text-sm text-slate-900 dark:text-slate-100 outline-none focus:ring-2 focus:ring-emerald-500/30"
              placeholder="至少 8 位"
              disabled={isSubmitting}
            />
          </div>

          {error && (
            <div className="rounded-xl border border-red-200 dark:border-red-900/60 bg-red-50 dark:bg-red-950/40 px-3 py-2 text-sm text-red-600 dark:text-red-300">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed text-white font-medium py-2.5 transition"
          >
            {isSubmitting ? "处理中..." : submitLabel}
          </button>
        </form>
      </div>
    </div>
  );
}
