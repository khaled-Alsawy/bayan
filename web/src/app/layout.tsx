import type { Metadata } from "next";
import { cookies } from "next/headers";

import { getMessages, isUiLocale, localeDirection, UI_LOCALE_COOKIE } from "@/lib/i18n";

import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const savedLocale = (await cookies()).get(UI_LOCALE_COOKIE)?.value;
  const locale = isUiLocale(savedLocale) ? savedLocale : "ar";
  return {
    title: getMessages(locale).metaTitle,
    description:
      locale === "ar"
        ? "مساحة بحث قرآني موثَّق وشفاف المصادر."
        : "A source-transparent Quran research workspace.",
  };
}

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const savedLocale = (await cookies()).get(UI_LOCALE_COOKIE)?.value;
  const locale = isUiLocale(savedLocale) ? savedLocale : "ar";

  return (
    <html lang={locale} dir={localeDirection(locale)}>
      <body>{children}</body>
    </html>
  );
}
