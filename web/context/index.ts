/**
 * Context exports - central export point for all contexts
 */

// Composite provider for app layout
export { CompositeProvider } from "./CompositeProvider";

// Individual context hooks
export { useQuestion, QuestionProvider } from "./question";
export { useResearch, ResearchProvider } from "./research";
export { useChat, ChatProvider } from "./chat";
export { useUISettings, UISettingsProvider } from "./settings";
export { useSidebar, SidebarProvider } from "./settings";

// Re-export types
export type { SidebarNavOrder } from "./settings";

// Legacy compatibility - useGlobal hook that combines all contexts
// This allows gradual migration without breaking existing code
export { useGlobal, GlobalProvider } from "./GlobalContext";
