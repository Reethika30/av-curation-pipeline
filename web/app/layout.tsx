import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AV Curation Pipeline",
  description:
    "Multimodal embedding-based curation, near-duplicate detection, failure-mode clustering and lineage tracking for autonomous-driving datasets.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-mono antialiased">{children}</body>
    </html>
  );
}
