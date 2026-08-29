import type { Metadata } from "next";

import { VoiceApp } from "@/components/voice-app";

/**
 * Every saved thread is somebody's, and the id in the URL is the only thing
 * guarding it — so the title says nothing about what is inside and the page
 * stays out of every index.
 */
export const metadata: Metadata = {
  title: "Conversation",
  description: "A saved voice conversation, reopened where it left off.",
  robots: { index: false, follow: false, nocache: true },
};


/**
 * One conversation, at its own address.
 *
 * The same screen as `/`, with two differences the id buys: the socket opens
 * against this conversation, so the model is handed back what it already said,
 * and the panels load the thread out of Postgres instead of starting empty.
 *
 * An id that is junk, deleted, or someone else's is not an error here. The
 * socket declines to bind it, the thread loads empty, and the next thing
 * spoken opens a fresh conversation — which is a better answer to a stale
 * bookmark than a 404 in front of a microphone.
 */
export default async function ConversationPage({ params }: PageProps<"/c/[id]">) {
  const { id } = await params;

  return (
    <main className="stage min-h-dvh flex-1">
      <VoiceApp conversationId={id} />
    </main>
  );
}
