import { cookies } from "next/headers";

import { getMessages, isUiLocale, UI_LOCALE_COOKIE } from "@/lib/i18n";

export default async function TermsPage() {
  const savedLocale = (await cookies()).get(UI_LOCALE_COOKIE)?.value;
  const copy = getMessages(isUiLocale(savedLocale) ? savedLocale : "ar");
  return <main className="mx-auto max-w-3xl p-8 leading-8"><h1>{copy.terms.title}</h1><p>{copy.terms.introduction}</p><p>{copy.terms.provenance}</p><p>{copy.terms.use}</p></main>;
}
