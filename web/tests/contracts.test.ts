import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  ANSWER_STATUSES,
  ResearchRequestSchema,
  ResearchResponseSchema,
} from "@/lib/contracts";
import type { RequestContractParity, ResponseContractParity } from "@/lib/contracts";
import { ARABIC_EXACT, makeResearchResponse } from "./fixtures";

describe("canonical network contracts", () => {
  it("enforces bidirectional full-shape OpenAPI parity at compile time", () => {
    const requestParity: RequestContractParity = [true, true];
    const responseParity: ResponseContractParity = [true, true];
    expect(requestParity).toEqual([true, true]);
    expect(responseParity).toEqual([true, true]);
  });

  it("matches the AnswerStatus enum exported by FastAPI OpenAPI", () => {
    const schema = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "openapi.json"), "utf8"),
    ) as { components: { schemas: { AnswerStatus: { enum: string[] } } } };
    expect(schema.components.schemas.AnswerStatus.enum).toEqual([...ANSWER_STATUSES]);
  });

  it("preserves exact source text byte-for-byte through runtime validation", () => {
    const parsed = ResearchResponseSchema.parse(makeResearchResponse());
    const source = parsed.blocks.find((block) => block.block_type === "SOURCE");
    expect(source?.block_type === "SOURCE" ? source.exact_text : null).toBe(ARABIC_EXACT);
  });

  it("rejects unknown response fields and malformed citations", () => {
    const malformed = {
      ...makeResearchResponse(),
      internal_trace: { provider: "must-not-leak" },
    };
    expect(ResearchResponseSchema.safeParse(malformed).success).toBe(false);

    const missingProvenance = structuredClone(makeResearchResponse()) as Record<string, unknown>;
    const citations = missingProvenance.citations as Array<Record<string, unknown>>;
    delete citations[0].source_revision;
    expect(ResearchResponseSchema.safeParse(missingProvenance).success).toBe(false);
  });

  it("bounds and strictly validates browser requests", () => {
    expect(ResearchRequestSchema.safeParse({ message: "a", locale: "en" }).success).toBe(false);
    expect(
      ResearchRequestSchema.safeParse({ message: "patience", locale: "en", provider: "groq" })
        .success,
    ).toBe(false);
  });
});
