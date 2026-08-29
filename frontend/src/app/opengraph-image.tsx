import { ImageResponse } from "next/og";

/**
 * The card a link to this app unfurls into.
 *
 * Drawn rather than shipped as a PNG: the screenshot in `images/home.png` is
 * mostly the empty white room the orb sits in, which reads as a broken image
 * at the 300px a timeline actually renders. This is the same room cropped to
 * what survives that size — the mark, the claim, and the two names doing the
 * work.
 *
 * `alt`, `size` and `contentType` below are the metadata Next emits beside the
 * image; `twitter:image` picks the same file up, which is why there is no
 * `twitter-image.tsx` next to this one.
 */
export const alt = "Vec — speak, and be answered";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

/** The orb's waveform, tallest in the middle — nine bars, as on screen. */
const BARS = [56, 104, 150, 196, 232, 196, 150, 104, 56];

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          display: "flex",
          width: "100%",
          height: "100%",
          position: "relative",
          backgroundColor: "#ffffff",
          color: "#101014",
        }}
      >
        {/*
          The washes from `--ambient`. Satori has no multi-layer backgrounds,
          so each one is its own absolutely positioned sheet — without them the
          card is flat white and the orb has nothing to sit in. Sized rather
          than `inset: 0`: satori reads the longhands only, and the shorthand
          silently leaves every sheet zero-sized.
        */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: 1200,
            height: 630,
            display: "flex",
            backgroundImage:
              "radial-gradient(circle at 6% 0%, rgba(106,124,166,0.30), rgba(255,255,255,0) 55%)",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: 1200,
            height: 630,
            display: "flex",
            backgroundImage:
              "radial-gradient(circle at 100% 4%, rgba(178,146,122,0.28), rgba(255,255,255,0) 50%)",
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: 1200,
            height: 630,
            display: "flex",
            backgroundImage:
              "radial-gradient(circle at 78% 100%, rgba(118,154,146,0.26), rgba(255,255,255,0) 55%)",
          }}
        />

        <div
          style={{
            display: "flex",
            width: "100%",
            height: "100%",
            padding: "72px 80px",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", maxWidth: 610 }}>
            <div style={{ display: "flex", alignItems: "center", marginBottom: 38 }}>
              <div
                style={{
                  display: "flex",
                  width: 52,
                  height: 52,
                  borderRadius: 26,
                  backgroundColor: "#101014",
                  color: "#ffffff",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 28,
                  fontWeight: 700,
                }}
              >
                V
              </div>
              <div
                style={{
                  marginLeft: 18,
                  fontSize: 36,
                  fontWeight: 600,
                  letterSpacing: "-0.01em",
                }}
              >
                Vec
              </div>
            </div>

            <div
              style={{
                display: "flex",
                fontSize: 66,
                fontWeight: 600,
                lineHeight: 1.08,
                letterSpacing: "-0.035em",
              }}
            >
              Speak, and be answered.
            </div>

            <div
              style={{
                display: "flex",
                marginTop: 26,
                fontSize: 27,
                lineHeight: 1.45,
                color: "#4a4a54",
              }}
            >
              Twenty-two Indian languages, heard and read back out loud in the one
              you spoke — grounded in the vector store you connect.
            </div>

            <div style={{ display: "flex", marginTop: 44 }}>
              {["Sarvam Saaras hears", "Bulbul answers", "1.9s to first audio"].map(
                (chip) => (
                  <div
                    key={chip}
                    style={{
                      display: "flex",
                      marginRight: 14,
                      padding: "10px 20px",
                      borderRadius: 999,
                      border: "1px solid rgba(16,16,20,0.14)",
                      backgroundColor: "rgba(255,255,255,0.7)",
                      fontSize: 21,
                      color: "#4a4a54",
                    }}
                  >
                    {chip}
                  </div>
                ),
              )}
            </div>
          </div>

          <div
            style={{
              display: "flex",
              width: 340,
              height: 340,
              borderRadius: 170,
              backgroundColor: "#ffffff",
              border: "1px solid rgba(16,16,20,0.08)",
              boxShadow: "0 40px 90px -40px rgba(16,16,20,0.45)",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {BARS.map((height, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  width: 20,
                  height,
                  marginLeft: i === 0 ? 0 : 10,
                  borderRadius: 10,
                  backgroundColor: "#101014",
                }}
              />
            ))}
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
