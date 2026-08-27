import { ClerkProvider } from "@clerk/nextjs";
import { shadcn } from "@clerk/ui/themes";
import type { Metadata, Viewport } from "next";
import { Fraunces, Manrope, Noto_Sans_Devanagari } from "next/font/google";
import "./globals.css";

import { AppHeader } from "@/components/app-header";
import { AppSidebar } from "@/components/app-sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ConversationProvider } from "@/lib/conversation";

/** Interface voice: rounded, geometric, calm at the small sizes the rail uses. */
const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
});

/**
 * Display voice. Fraunces reads like an ordinary text serif until its SOFT,
 * WONK and opsz axes are pushed — the character lives in `.type-display` and
 * `.type-quote` in globals.css, not here. Those axes have to be requested or
 * the file ships weight-only and the settings do nothing.
 */
const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  axes: ["SOFT", "WONK", "opsz"],
});

/**
 * Nine of the twenty-two languages Vec speaks are written in Devanagari,
 * and no Latin face covers a single one of its glyphs — without this the
 * transcript silently drops to whatever the OS picks. Not preloaded: most
 * takes never need it, and the browser fetches it the moment one does.
 *
 * The other Indic scripts still fall through to the system default. Bundling
 * a Noto per script would cost far more than it returns.
 */
const devanagari = Noto_Sans_Devanagari({
  variable: "--font-devanagari",
  subsets: ["devanagari"],
  preload: false,
});

export const metadata: Metadata = {
  title: "Vec — Speak, and be answered",
  description:
    "A voice interface that listens in any Indian language and answers out loud in the same one, with Sarvam Saaras and Bulbul.",
};

export const viewport: Viewport = {
  themeColor: "#ffffff",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${manrope.variable} ${fraunces.variable} ${devanagari.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <ClerkProvider appearance={{ theme: shadcn }}>
          <ConversationProvider>
            <TooltipProvider delayDuration={150}>
              <AppHeader />
              <AppSidebar />
              {children}
              {/* pb bumps up under ~770px, where the centered line would
                  otherwise run under the bottom-right rail. */}
              <footer className="pointer-events-none fixed inset-x-0 bottom-0 z-30 px-6 py-5 text-center text-[0.72rem] tracking-wide text-ink-muted max-md:pb-24">
                Sarvam Saaras hears · Bulbul answers · 30 seconds per take
              </footer>
            </TooltipProvider>
          </ConversationProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}