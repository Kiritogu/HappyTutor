"use client";

import { FormEvent, useMemo, useState } from "react";
import Image from "next/image";
import { useAuth } from "@/context/AuthContext";

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

type Mode = "login" | "register";
const TRUST_POINTS = [
  "多轮对话与检索增强，回答更可靠",
  "学习记录与知识资产按账号隔离",
  "从提问到复盘，一体化学习流",
];

export default function LoginPage() {
  const { login, register } = useAuth();

  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const title = useMemo(
    () => (mode === "login" ? "欢迎回来" : "创建账号"),
    [mode],
  );
  const subtitle = useMemo(
    () =>
      mode === "login"
        ? "登录你的智学助手账号"
        : "注册一个新账号开始学习",
    [mode],
  );
  const submitLabel = mode === "login" ? "登录" : "注册并登录";

  const validate = (): string | null => {
    if (!email.trim()) return "请输入邮箱";
    if (!EMAIL_REGEX.test(email.trim())) return "邮箱格式不正确";
    if (!password) return "请输入密码";
    if (password.length < 8) return "密码至少 8 位";
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
      setError(
        submitError instanceof Error
          ? submitError.message
          : "请求失败，请稍后重试",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#f5f4f0] text-slate-900 [font-family:'Avenir_Next','PingFang_SC','Microsoft_YaHei',sans-serif] dark:bg-[#121418] dark:text-slate-100">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-28 right-[-110px] h-96 w-96 rounded-full bg-emerald-300/30 blur-3xl dark:bg-emerald-500/20" />
        <div className="absolute -bottom-28 -left-24 h-80 w-80 rounded-full bg-slate-300/60 blur-3xl dark:bg-slate-700/35" />
        <div
          className="absolute inset-0 opacity-35 dark:opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle at 1px 1px, rgba(15,23,42,0.08) 1px, transparent 0)",
            backgroundSize: "24px 24px",
          }}
        />
      </div>

      <div className="relative flex min-h-screen items-center justify-center px-5 py-10 sm:px-8">
        <div className="w-full max-w-5xl overflow-hidden rounded-[30px] border border-slate-300/70 bg-white/85 shadow-[0_40px_100px_-60px_rgba(15,23,42,0.75)] backdrop-blur-xl dark:border-slate-700/60 dark:bg-slate-900/75 animate-[fade-in_0.7s_ease-out]">
          <div className="grid md:grid-cols-[1.05fr_1fr]">
            <section className="hidden border-r border-slate-200/80 p-10 md:flex md:flex-col md:justify-between dark:border-slate-800/80">
              <div>
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-xl bg-slate-900 dark:bg-slate-100">
                    <Image
                      src="/logo.png"
                      alt="智学助手"
                      width={24}
                      height={24}
                      className="object-contain"
                      priority
                    />
                  </div>
                  <span className="text-sm tracking-[0.2em] text-slate-500 dark:text-slate-400">
                    ZHI XUE ASSISTANT
                  </span>
                </div>

                <h2 className="mt-14 text-4xl leading-tight text-slate-900 [font-family:'Iowan_Old_Style','Noto_Serif_SC','Songti_SC',serif] dark:text-slate-100">
                  让学习回到
                  <br />
                  专注本身
                </h2>
                <p className="mt-5 max-w-sm text-sm leading-7 text-slate-500 dark:text-slate-400">
                  为长期学习设计的 AI 空间。少干扰，高可信，信息与思考都被妥善保存。
                </p>
              </div>

              <div className="space-y-4">
                {TRUST_POINTS.map((point) => (
                  <div
                    key={point}
                    className="flex items-start gap-3 text-sm text-slate-600 dark:text-slate-300"
                  >
                    <span className="mt-2 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-emerald-600 dark:bg-emerald-400" />
                    <span>{point}</span>
                  </div>
                ))}
                <p className="pt-3 text-xs tracking-[0.12em] text-slate-400 dark:text-slate-500">
                  PRIVATE · RELIABLE · FOCUSED
                </p>
              </div>
            </section>

            <section className="p-7 sm:p-10 md:p-12">
              <div className="mb-9 md:hidden">
                <div className="mb-6 flex items-center gap-2.5">
                  <div className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-lg bg-slate-900 dark:bg-slate-100">
                    <Image
                      src="/logo.png"
                      alt="智学助手"
                      width={22}
                      height={22}
                      className="object-contain"
                      priority
                    />
                  </div>
                  <span className="text-base tracking-[0.14em] text-slate-600 dark:text-slate-300">
                    智学助手
                  </span>
                </div>
              </div>

              <div className="mb-7">
                <h1 className="text-3xl leading-tight text-slate-900 [font-family:'Iowan_Old_Style','Noto_Serif_SC','Songti_SC',serif] dark:text-slate-100">
                  {title}
                </h1>
                <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                  {subtitle}
                </p>
              </div>

              <div className="mb-8 flex gap-5 border-b border-slate-200 pb-3 dark:border-slate-700">
                <button
                  type="button"
                  onClick={() => setMode("login")}
                  className={`relative pb-1 text-sm transition-colors ${
                    mode === "login"
                      ? "text-slate-900 dark:text-slate-100"
                      : "text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300"
                  }`}
                >
                  登录
                  <span
                    className={`absolute bottom-0 left-0 h-px bg-slate-900 transition-all dark:bg-slate-100 ${
                      mode === "login" ? "w-full" : "w-0"
                    }`}
                  />
                </button>
                <button
                  type="button"
                  onClick={() => setMode("register")}
                  className={`relative pb-1 text-sm transition-colors ${
                    mode === "register"
                      ? "text-slate-900 dark:text-slate-100"
                      : "text-slate-400 hover:text-slate-600 dark:text-slate-500 dark:hover:text-slate-300"
                  }`}
                >
                  注册
                  <span
                    className={`absolute bottom-0 left-0 h-px bg-slate-900 transition-all dark:bg-slate-100 ${
                      mode === "register" ? "w-full" : "w-0"
                    }`}
                  />
                </button>
              </div>

              <form onSubmit={onSubmit} className="space-y-6">
                <div>
                  <label className="mb-1 block text-xs tracking-[0.16em] text-slate-500 dark:text-slate-400">
                    邮箱
                  </label>
                  <input
                    type="email"
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full border-0 border-b border-slate-300 bg-transparent px-0 py-2.5 text-[15px] text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-slate-900 focus:ring-0 dark:border-slate-600 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-slate-100"
                    placeholder="you@example.com"
                    disabled={isSubmitting}
                  />
                </div>

                <div>
                  <label className="mb-1 block text-xs tracking-[0.16em] text-slate-500 dark:text-slate-400">
                    密码
                  </label>
                  <input
                    type="password"
                    autoComplete={
                      mode === "login" ? "current-password" : "new-password"
                    }
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full border-0 border-b border-slate-300 bg-transparent px-0 py-2.5 text-[15px] text-slate-900 outline-none transition-colors placeholder:text-slate-400 focus:border-slate-900 focus:ring-0 dark:border-slate-600 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-slate-100"
                    placeholder="至少 8 位"
                    disabled={isSubmitting}
                  />
                </div>

                {error && (
                  <div className="rounded-xl border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-600 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-300">
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full rounded-xl bg-slate-900 py-3 text-sm text-white transition-colors hover:bg-slate-800 active:bg-slate-950 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
                >
                  {isSubmitting ? "处理中..." : submitLabel}
                </button>
              </form>

              <p className="mt-8 text-center text-xs tracking-[0.08em] text-slate-400 dark:text-slate-500">
                使用邮箱登录，所有数据按账号严格隔离
              </p>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
