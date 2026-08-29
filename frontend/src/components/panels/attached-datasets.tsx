"use client";

import { AlertCircle, Loader2, Plus, RefreshCw, Table2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { PanelChip, PanelEmpty } from "@/components/panels/panel";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  addDataset,
  DatasetError,
  describeDataset,
  listDatasets,
  rebuildDataset,
  removeDataset,
  type Dataset,
} from "@/lib/datasets";
import { cn } from "@/lib/utils";

/**
 * Six datasets to start from, for somebody staring at an empty URL field.
 *
 * The same move `TryAsking` makes on the orb: a person who has never attached
 * one does not know what shape of URL is wanted, let alone that a bare
 * `owner/name` is enough — `normalise()` in `source.py` accepts it, so that is
 * what these send. Tapping one is the shortest proof that the thing works.
 *
 * Picked to be *answerable in SQL* rather than merely famous: every one is
 * public, ungated, and already converted to parquet or CSV on Hugging Face, so
 * none of them fails at pull time on a person's first click. MNIST and the
 * speech corpora are more popular still and are columns of image and audio
 * bytes — nothing to count, nothing to group by. Aya holds the last slot on
 * theme rather than on downloads: it is the one here with 65 languages in it,
 * which is the thing this app is for.
 */
const POPULAR: { repo: string; label: string; note: string }[] = [
  {
    repo: "fka/awesome-chatgpt-prompts",
    label: "ChatGPT prompts",
    note: "a few hundred prompt patterns, one CSV",
  },
  {
    repo: "stanfordnlp/imdb",
    label: "IMDb reviews",
    note: "50,000 labelled movie reviews",
  },
  {
    repo: "rajpurkar/squad",
    label: "SQuAD",
    note: "98,000 questions over Wikipedia paragraphs",
  },
  {
    repo: "openai/gsm8k",
    label: "GSM8K",
    note: "8,500 grade-school maths problems",
  },
  {
    repo: "tatsu-lab/alpaca",
    label: "Alpaca",
    note: "52,000 instruction-and-response pairs",
  },
  {
    repo: "CohereLabs/aya_dataset",
    label: "Aya",
    note: "204,000 human prompts across 65 languages",
  },
];

/**
 * What somebody does *after* opening the Dataset connector: attach several.
 *
 * The second connector with a second step, and for the same reason as
 * `ComposioToolkits` — one credential is a doorway rather than a destination.
 * A person has one Pinecone and many datasets, so the connector row records
 * that datasets are attached and this records which.
 *
 * Attaching returns immediately with `pending`, because the pull, the
 * measurement and the model call all run on a worker. So this polls while
 * anything is building, and stops the moment nothing is — the same shape the
 * Composio consent screen needs, arrived at from the other direction.
 */
export function AttachedDatasets() {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [limit, setLimit] = useState(0);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  // Which URL is in flight, not merely *that* one is: six chips and a form
  // all share this, and the spinner has to land on the one that was tapped.
  const [adding, setAdding] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [reloads, setReloads] = useState(0);
  const reload = useCallback(() => setReloads((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();

    void listDatasets(controller.signal).then((body) => {
      if (controller.signal.aborted) return;
      setDatasets(body.datasets);
      setLimit(body.limit);
    });

    return () => controller.abort();
  }, [reloads]);

  const building = useMemo(
    () => (datasets ?? []).some((dataset) => dataset.status === "pending"),
    [datasets],
  );

  // A build is seconds to a minute — a 55 GB repo took 30 s to sample. Poll
  // while one is running and stop the moment none is, so an idle panel is not
  // a request every three seconds forever.
  useEffect(() => {
    if (!building) return;
    const timer = setInterval(reload, 3_000);
    return () => clearInterval(timer);
  }, [building, reload]);

  const attached = datasets ?? [];
  const full = limit > 0 && attached.length >= limit;

  // A chip for something already in the list below is a click that returns an
  // error, so an attached dataset drops out of the row. Matched against both
  // `location` (the bare repo, for an HF one) and `url` (what was actually
  // sent), because a person may have pasted the full link for the same thing.
  const suggestions = useMemo(() => {
    const taken = (datasets ?? []).map((d) => `${d.location} ${d.url}`.toLowerCase());
    return POPULAR.filter(
      (p) => !taken.some((entry) => entry.includes(p.repo.toLowerCase())),
    );
  }, [datasets]);

  const attach = async (value: string) => {
    if (!value || adding) return;

    setAdding(value);
    setError(null);
    try {
      await addDataset(value);
      // Only if this is what the field holds. A chip attaching `openai/gsm8k`
      // must not wipe a URL somebody is halfway through typing.
      setUrl((current) => (current.trim() === value ? "" : current));
      reload();
    } catch (raised) {
      // The message is written for whoever typed the URL — "No dataset called
      // x/y", "gated or private" — so it is shown rather than replaced.
      setError(raised instanceof DatasetError ? raised.message : "Could not attach that.");
    } finally {
      setAdding(null);
    }
  };

  const add = (event: React.FormEvent) => {
    event.preventDefault();
    void attach(url.trim());
  };

  const detach = async (datasetId: string) => {
    setBusy(datasetId);
    setError(null);
    try {
      await removeDataset(datasetId);
      reload();
    } catch (raised) {
      setError(raised instanceof DatasetError ? raised.message : "Could not remove that.");
    } finally {
      setBusy(null);
    }
  };

  const refresh = async (datasetId: string) => {
    setBusy(datasetId);
    setError(null);
    try {
      await rebuildDataset(datasetId);
      reload();
    } catch (raised) {
      setError(raised instanceof DatasetError ? raised.message : "Could not rebuild that.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <form onSubmit={add} className="flex items-center gap-1.5">
        {/* the <label> is the field, which is why `.glass-field` lights on
            `focus-within` rather than on `focus` */}
        <label className="glass-field flex min-w-0 flex-1 items-center gap-2 rounded-lg px-2.5 py-1.5">
          <Table2 aria-hidden className="size-3.5 shrink-0 text-ink-muted" />
          <input
            type="url"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            disabled={full}
            placeholder={
              full ? `${limit} datasets is the limit` : "huggingface.co/datasets/owner/name"
            }
            className="min-w-0 flex-1 bg-transparent text-[0.75rem] text-ink outline-none placeholder:text-ink-muted disabled:cursor-not-allowed"
          />
        </label>

        <Button
          type="submit"
          size="sm"
          disabled={!url.trim() || adding !== null || full}
        >
          {adding === url.trim() ? (
            <Loader2 aria-hidden className="size-3.5 animate-spin" />
          ) : (
            <Plus aria-hidden className="size-3.5" />
          )}
          <span className="sr-only">Attach dataset</span>
        </Button>
      </form>

      {error && (
        <p className="flex items-start gap-1.5 px-1 text-[0.7rem] text-danger">
          <AlertCircle aria-hidden className="mt-px size-3 shrink-0" />
          {error}
        </p>
      )}

      {datasets === null ? (
        <PanelEmpty icon={Loader2}>Loading…</PanelEmpty>
      ) : attached.length === 0 ? (
        <PanelEmpty icon={Table2}>
          Nothing attached yet. Paste a Hugging Face dataset, or a link to a .parquet or
          .csv file.
        </PanelEmpty>
      ) : (
        <ScrollArea className="max-h-56">
          <div className="flex flex-col gap-0.5 pr-2">
            {attached.map((dataset) => (
              <Row
                key={dataset.datasetId}
                dataset={dataset}
                busy={busy === dataset.datasetId}
                onRemove={() => detach(dataset.datasetId)}
                onRebuild={() => refresh(dataset.datasetId)}
              />
            ))}
          </div>
        </ScrollArea>
      )}

      {suggestions.length > 0 && !full && (
        <section aria-label="Popular datasets" className="flex flex-col gap-1.5">
          <p className="px-1 text-[0.66rem] text-ink-muted">
            Or attach a popular one
          </p>

          {/*
            A pill is one line by construction, so the label is the short human
            name and `title` carries the repo and what is in it. The repo is
            what gets sent: `normalise()` turns a bare `owner/name` into the
            full Hugging Face URL, so the chip does not have to spell one out.
          */}
          <ul className="flex flex-wrap gap-1.5 px-0.5">
            {suggestions.map((suggestion) => (
              <li key={suggestion.repo} className="max-w-full">
                <button
                  type="button"
                  disabled={adding !== null}
                  onClick={() => void attach(suggestion.repo)}
                  title={`${suggestion.repo} — ${suggestion.note}`}
                  className="glass glass-hover flex max-w-full items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.72rem] text-ink-soft disabled:pointer-events-none disabled:opacity-40"
                >
                  {adding === suggestion.repo ? (
                    <Loader2 aria-hidden className="size-3 shrink-0 animate-spin" />
                  ) : (
                    <Plus aria-hidden className="size-3 shrink-0 text-ink-muted" />
                  )}
                  <span className="truncate">{suggestion.label}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {limit > 0 && attached.length > 0 && (
        <p className="px-1 text-[0.66rem] text-ink-muted">
          {attached.length} of {limit} attached
        </p>
      )}
    </div>
  );
}

function Row({
  dataset,
  busy,
  onRemove,
  onRebuild,
}: {
  dataset: Dataset;
  busy: boolean;
  onRemove: () => void;
  onRebuild: () => void;
}) {
  const pending = dataset.status === "pending";
  const failed = dataset.status === "failed";

  // The failure reason, not the status word. "could not read x/y on Hugging
  // Face" is actionable; "failed" sends somebody to the logs.
  const detail = failed
    ? dataset.error || "Could not be built"
    : pending
      ? "Pulling and measuring…"
      : describeDataset(dataset) || dataset.location;

  return (
    <div className="glass-row flex items-center gap-2.5 rounded-lg px-2 py-2">
      <span className="glass-tile grid size-7 shrink-0 place-items-center rounded-lg text-ink-muted">
        {pending ? (
          <Loader2 aria-hidden className="size-3.5 animate-spin" />
        ) : (
          <Table2 aria-hidden className="size-3.5" />
        )}
      </span>

      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[0.8rem] font-medium text-ink">
            {dataset.title || dataset.datasetId}
          </span>
          {dataset.status === "degraded" && <PanelChip>partial</PanelChip>}
        </span>
        <span
          className={cn(
            "truncate text-[0.7rem]",
            failed ? "text-danger" : "text-ink-muted",
          )}
        >
          {detail}
        </span>
      </span>

      {!pending && (
        <button
          type="button"
          onClick={onRebuild}
          disabled={busy}
          title="Pull and measure it again"
          className="shrink-0 rounded p-1 text-ink-muted transition-colors hover:text-ink disabled:opacity-50"
        >
          <RefreshCw aria-hidden className="size-3.5" />
          <span className="sr-only">Rebuild {dataset.datasetId}</span>
        </button>
      )}

      <button
        type="button"
        onClick={onRemove}
        disabled={busy}
        title="Detach and delete the local copy"
        className="shrink-0 rounded p-1 text-ink-muted transition-colors hover:text-danger disabled:opacity-50"
      >
        {busy ? (
          <Loader2 aria-hidden className="size-3.5 animate-spin" />
        ) : (
          <X aria-hidden className="size-3.5" />
        )}
        <span className="sr-only">Remove {dataset.datasetId}</span>
      </button>
    </div>
  );
}
