import { cookies } from "next/headers";

import { ResearchWorkspace } from "@/components/research-workspace";
import { isUiLocale, UI_LOCALE_COOKIE } from "@/lib/i18n";

export default async function Home() {
  const savedLocale = (await cookies()).get(UI_LOCALE_COOKIE)?.value;
  const locale = isUiLocale(savedLocale) ? savedLocale : "ar";

  return <ResearchWorkspace initialLocale={locale} />;
}
