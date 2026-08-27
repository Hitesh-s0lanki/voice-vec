import { SignIn } from "@clerk/nextjs";

export default function SignInPage() {
  return (
    // `.stage` is what gives Clerk's card something to refract — it is a
    // backdrop root, so its own `backdrop-filter` samples the washes painted
    // here. On a bare `<div>` the card blurs the page background and reads as
    // flat white. The glass itself comes from the `.cl-*` block in globals.css.
    <main className="stage flex min-h-dvh flex-1 items-center justify-center px-6 py-16">
      <SignIn />
    </main>
  );
}
