import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://mint-agent-evidence-desk.wang-kun221803.chatgpt.site"),
  title: "Mint Agent — Evidence-first Financial Research",
  description:
    "Ask a financial question, follow the live research timeline, and inspect every source behind the answer.",
  openGraph: {
    title: "Mint Agent — Evidence-first Financial Research",
    description:
      "Ask a financial question, follow the live research timeline, and inspect every source behind the answer.",
    images: [
      {
        url: "/og-finance-jade.png",
        width: 1536,
        height: 1024,
        alt: "Mint Agent financial research workspace",
      },
    ],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Mint Agent — Evidence-first Financial Research",
    description:
      "Ask a financial question, follow the live research timeline, and inspect every source behind the answer.",
    images: ["/og-finance-jade.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export const viewport: Viewport = {
  themeColor: "#f7f8f4",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>
        <a className="skip-link" href="#main-content">
          跳到主要内容 / Skip to main content
        </a>
        {children}
      </body>
    </html>
  );
}
