export default function SettingsPage() {
  return (
    <main className="stage flex min-h-dvh flex-1 flex-col items-center justify-center gap-4 px-6 py-20">
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
