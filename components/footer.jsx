import Link from "next/link"
import { Youtube, Mail } from "lucide-react"

export default function Footer() {
  return (
    <footer className="border-t border-gray-200 py-6">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center">
          <div className="font-medium text-xl">
            <span className="text-purple-700">AIE</span>
          </div>
          <div className="flex gap-4">
            <Link href="https://youtube.com" target="_blank" aria-label="YouTube">
              <Youtube className="h-5 w-5 text-gray-600 hover:text-purple-700 transition-colors" />
            </Link>
            <Link href="mailto:uscshiftscaie@gmail.com" aria-label="Email">
              <Mail className="h-5 w-5 text-gray-600 hover:text-purple-700 transition-colors" />
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}
