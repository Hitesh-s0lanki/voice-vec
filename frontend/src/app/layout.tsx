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

/**
 * Where this deployment answers from. Every absolute URL below — the canonical
 * link, the OG card's `url` and its image — is composed from it, and a
 * relative path in any of those fields is a build error without it. Unset it
 * and the metadata still resolves, just against localhost, which is right for
 * a dev box and wrong the moment a link is shared.
 */
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3002";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  /*
   * `default` is the home screen's title; `template` is what every other
   * segment gets wrapped in, so a page exports `title: "Settings"` and the tab
   * reads "Settings · Vec" without repeating the product name in five files.
   */
  title: {
    default: "Vec — speak, and be answered",
    template: "%s · Vec",
  },
  description:
    "A voice interface that hears you in any of twenty-two Indian languages and answers out loud in the same one, grounded in the vector store you connect.",
  applicationName: "Vec",
  keywords: [
    "voice assistant",
    "speech to text",
    "text to speech",
    "Sarvam AI",
    "Saaras",
    "Bulbul",
    "Indian languages",
    "multilingual RAG",
    "retrieval augmented generation",
    "pgvector",
  ],
  authors: [{ name: "Hitesh Solanki", url: "https://github.com/Hitesh-s0lanki" }],
  creator: "Hitesh Solanki",
  category: "technology",
  alternates: { canonical: "/" },
  // A phone number is 10 digits and so is a retrieval score; iOS turns the
  // second one into a tel: link if it is not told not to.
  formatDetection: { telephone: false, address: false, email: false },
  openGraph: {
    type: "website",
    siteName: "Vec",
    title: "Vec — speak, and be answered",
    description:
      "Speak in Hindi, Tamil, Kannada — any of twenty-two languages — and be answered out loud in the one you spoke.",
    url: "/",
    locale: "en_IN",
  },
  twitter: {
    // `summary_large_image` is the only card that shows opengraph-image.tsx at
    // the size it is drawn; `summary` crops it to a square thumbnail.
    card: "summary_large_image",
    title: "Vec — speak, and be answered",
    description:
      "Speak in Hindi, Tamil, Kannada — any of twenty-two languages — and be answered out loud in the one you spoke.",
  },
  // Installed to a home screen this is a full-bleed app, not a page: the black
  // orb wants the whole viewport, and `apple-icon.png` is already beside this
  // file for the icon.
  appleWebApp: { capable: true, title: "Vec", statusBarStyle: "default" },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true, "max-image-preview": "large" },
  },
};

export const viewport: Viewport = {
  themeColor: "#ffffff",
  // `cover` lets the stage's washes run under the notch and the home
  // indicator; everything that would otherwise land beneath one — the rail,
  // this footer — pads itself back out with `env(safe-area-inset-*)`.
  viewportFit: "cover",
  colorScheme: "light",
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
              {/* Below `lg` the rail is centred on the bottom edge, so the
                  line has to clear the whole rail rather than just its
                  corner — and clear the home indicator under it as well. */}
              <footer className="pointer-events-none fixed inset-x-0 bottom-0 z-30 px-4 py-5 text-center text-[0.72rem] tracking-wide text-ink-muted sm:px-6 max-lg:pb-[calc(6rem+env(safe-area-inset-bottom))]">
                Sarvam Saaras hears · Bulbul answers · 30 seconds per take
              </footer>
            </TooltipProvider>
          </ConversationProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}