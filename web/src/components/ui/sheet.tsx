"use client";

import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";

export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;
export const SheetClose = DialogPrimitive.Close;
export const SheetTitle = DialogPrimitive.Title;
export const SheetDescription = DialogPrimitive.Description;

type SheetSide = "bottom" | "start" | "end";

export function SheetContent({
  className,
  children,
  closeLabel = "Close",
  side = "bottom",
  ...props
}: React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
  closeLabel?: string;
  side?: SheetSide;
}) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-[rgba(13,29,25,.42)] backdrop-blur-[2px] data-[state=closed]:animate-[fade-out_.18s_ease-in] data-[state=open]:animate-[fade-in_.2s_ease-out] motion-reduce:animate-none" />
      <DialogPrimitive.Content
        className={cn(
          "fixed z-50 overflow-y-auto border border-[var(--border)] bg-[var(--paper)] px-5 pb-8 pt-5 motion-reduce:animate-none",
          side === "bottom" && "inset-x-0 bottom-0 max-h-[88dvh] rounded-t-[1.75rem] shadow-[0_-18px_60px_rgba(21,42,35,.18)] data-[state=closed]:animate-[sheet-down_.22s_ease-in] data-[state=open]:animate-[sheet-up_.28s_cubic-bezier(.2,.8,.2,1)]",
          side === "start" && "sheet-side sheet-side-start",
          side === "end" && "sheet-side sheet-side-end",
          className,
        )}
        {...props}
      >
        {side === "bottom" ? <div aria-hidden className="mx-auto mb-5 h-1 w-10 rounded-full bg-[var(--border-strong)]" /> : null}
        {children}
        <DialogPrimitive.Close className="absolute end-4 top-4 grid size-10 place-items-center rounded-full text-[var(--ink-soft)] hover:bg-[var(--surface-strong)] focus-visible:outline-none focus-visible:ring-3 focus-visible:ring-[var(--focus)]">
          <X aria-hidden size={18} />
          <span className="sr-only">{closeLabel}</span>
        </DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  );
}
