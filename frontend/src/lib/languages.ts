/**
 * Sarvam returns BCP-47-ish codes (`hi-IN`). Show people a language, not a code.
 * Anything unrecognised falls back to the raw code so nothing renders empty.
 */
const LANGUAGE_NAMES: Record<string, string> = {
  "as-IN": "Assamese",
  "bn-IN": "Bengali",
  "brx-IN": "Bodo",
  "doi-IN": "Dogri",
  "en-IN": "English",
  "gu-IN": "Gujarati",
  "hi-IN": "Hindi",
  "kn-IN": "Kannada",
  "kok-IN": "Konkani",
  "ks-IN": "Kashmiri",
  "mai-IN": "Maithili",
  "ml-IN": "Malayalam",
  "mni-IN": "Manipuri",
  "mr-IN": "Marathi",
  "ne-IN": "Nepali",
  "od-IN": "Odia",
  "or-IN": "Odia",
  "pa-IN": "Punjabi",
  "sa-IN": "Sanskrit",
  "sat-IN": "Santali",
  "sd-IN": "Sindhi",
  "ta-IN": "Tamil",
  "te-IN": "Telugu",
  "ur-IN": "Urdu",
};

export function languageName(code: string | null): string | null {
  if (!code) return null;
  return LANGUAGE_NAMES[code] ?? LANGUAGE_NAMES[code.toLowerCase()] ?? code;
}
