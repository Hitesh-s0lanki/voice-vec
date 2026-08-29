import Image from "next/image";
import Link from "next/link";

/**
 * Fixed, transparent bar. It stays pointer-transparent so it never steals
 * clicks from the orb underneath — only the wordmark takes input.
 */
export function AppHeader() {
  return (
    <header className="pointer-events-none fixed inset-x-0 top-0 z-40 flex items-center px-4 py-4 sm:px-6 sm:py-5">
      <Link
        href="/"
        className="glass-row pointer-events-auto -mx-2 flex items-center gap-2.5 rounded-full px-2 py-1 outline-none focus-visible:ring-2 focus-visible:ring-ink/30 focus-visible:ring-offset-4 focus-visible:ring-offset-canvas"
      >
        {/*
          The artwork is a dark circle inset in an opaque near-white square.
          Clipping to a circle and scaling past that margin drops the square
          entirely, so the mark sits on any ground rather than only on white.
        */}
        <span aria-hidden className="block size-5 overflow-hidden rounded-full">
          <Image
            src="/logo.png"
            alt=""
            width={40}
            height={40}
            priority
            className="size-full scale-[1.2] object-cover"
          />
        </span>
        <span className="type-display text-[1.05rem] font-medium text-ink">
          Vec
        </span>
      </Link>
    </header>
  );
}
