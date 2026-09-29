import { describe, expect, it } from "vitest";

import { formatSurahReference, SURAH_NAMES } from "@/lib/surah-names";

describe("surah names", () => {
  it("contains all 114 surahs and the expected first display names", () => {
    expect(Object.keys(SURAH_NAMES).map(Number).sort((a, b) => a - b)).toEqual(Array.from({ length: 114 }, (_, index) => index + 1));
    expect(SURAH_NAMES[1]).toEqual({ arabic: "الفاتحة", english: "Al-Fatiha" });
    expect(SURAH_NAMES[2]).toEqual({ arabic: "البقرة", english: "Al-Baqarah" });
    expect(SURAH_NAMES[113]).toEqual({ arabic: "الفلق", english: "Al-Falaq" });
    expect(SURAH_NAMES[114]).toEqual({ arabic: "الناس", english: "An-Nas" });
  });

  it("formats full passage references and short citation labels", () => {
    expect(formatSurahReference("2:107-111", "ar")).toBe("سورة البقرة ١٠٧–١١١");
    expect(formatSurahReference("2:107-111", "en")).toBe("Al-Baqarah 107–111");
    expect(formatSurahReference("113:5", "ar", "short")).toBe("الفلق ٥");
    expect(formatSurahReference("113:5", "en", "short")).toBe("Al-Falaq 5");
  });
});
