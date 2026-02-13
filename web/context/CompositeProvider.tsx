"use client";

import React from "react";
import { QuestionProvider } from "./question";
import { ResearchProvider } from "./research";
import { ChatProvider } from "./chat";
import { UISettingsProvider, SidebarProvider } from "./settings";

/**
 * CompositeProvider combines all context providers into a single component.
 * This simplifies the provider hierarchy in the app layout.
 */
export function CompositeProvider({ children }: { children: React.ReactNode }) {
  return (
    <UISettingsProvider>
      <SidebarProvider>
        <QuestionProvider>
          <ResearchProvider>
            <ChatProvider>{children}</ChatProvider>
          </ResearchProvider>
        </QuestionProvider>
      </SidebarProvider>
    </UISettingsProvider>
  );
}
