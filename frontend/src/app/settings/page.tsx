import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Settings",
  description:
    "Tune capture length, language hints, and playback preferences for the voice loop.",
  // Signed-in surface: indexing it puts a login wall in search results.
  robots: { index: false, follow: false },
};

export default function SettingsPage() {
  return (
    <main className="stage flex min-h-dvh flex-1 flex-col items-center justify-center gap-4 px-4 py-24 sm:px-6">
      <div className="glass rise flex w-full max-w-md flex-col items-center gap-3 rounded-2xl px-8 py-9">
        <span aria-hidden className="bead size-5 rounded-full" />
        <h1 className="text-2xl font-medium tracking-[-0.02em] text-ink">
          Settings
        </h1>
        <p className="max-w-sm text-center text-[0.9rem] leading-relaxed text-ink-muted">
          Tune capture length, language hints, and playback preferences.
        </p>
      </div>
    </main>
  );
}
