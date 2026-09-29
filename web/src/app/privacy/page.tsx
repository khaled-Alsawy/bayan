import { cookies } from "next/headers";

import { getMessages, isUiLocale, UI_LOCALE_COOKIE } from "@/lib/i18n";

export default async function PrivacyPage() {
  const savedLocale = (await cookies()).get(UI_LOCALE_COOKIE)?.value;
  const copy = getMessages(isUiLocale(savedLocale) ? savedLocale : "ar");
  return <main className="mx-auto max-w-3xl p-8 leading-8"><h1>{copy.privacy.title}</h1><p>{copy.privacy.introduction}</p><h2>{copy.privacy.providersTitle}</h2><p>{copy.privacy.providers}</p><p>{copy.privacy.caution}</p></main>;
}
