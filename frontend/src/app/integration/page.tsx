export default function IntegrationPage() {
  return (
    <main className="stage flex min-h-dvh flex-1 flex-col items-center justify-center gap-4 px-6 py-20">
      <span aria-hidden className="bead size-5 rounded-full" />
      <h1 className="text-2xl font-medium tracking-[-0.02em] text-ink">
        Integration
      </h1>
      <p className="max-w-sm text-center text-[0.9rem] leading-relaxed text-ink-muted">
        Connect Vec to the services that should receive your transcripts.
      </p>
    </main>
  );
}
