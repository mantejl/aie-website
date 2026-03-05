"use client"
import Link from "next/link"
import { useEffect, useState } from "react"
import { usePathname } from "next/navigation"

export default function Navbar() {
  const pathname = usePathname()
  const isHomePage = pathname === "/"
  const [isLoaded, setIsLoaded] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  return (
    <nav
      className={`sticky top-3 z-50 w-full flex justify-center px-3 mt-4 bg-transparent transition-all duration-1000 ${
        isHomePage && !isLoaded ? "opacity-0 translate-y-10" : "opacity-100 translate-y-0"
      }`}
    >
      <div className="w-full max-w-3xl bg-white rounded-2xl shadow-lg px-4 py-3 flex flex-col md:flex-row md:items-center">
        {/* Desktop navigation */}
        <div className="hidden md:flex flex-1 items-center justify-center gap-8 lg:gap-12">
          <Link
            href="/"
            className="font-normal text-sm lg:text-base hover:text-purple-700 transition-transform duration-200 hover:scale-110"
          >
            Home
          </Link>
          <Link
            href="/work"
            className="font-normal text-sm lg:text-base hover:text-purple-700 transition-transform duration-200 hover:scale-110"
          >
            Work
          </Link>
          <Link
            href="/curriculum"
            className="font-normal text-sm lg:text-base hover:text-purple-700 transition-transform duration-200 hover:scale-110"
          >
            Curriculum
          </Link>
          <Link
            href="/team"
            className="font-normal text-sm lg:text-base hover:text-purple-700 transition-transform duration-200 hover:scale-110"
          >
            Team
          </Link>
          <Link
            href="/culture"
            className="font-normal text-sm lg:text-base hover:text-purple-700 transition-transform duration-200 hover:scale-110"
          >
            Culture
          </Link>
          <Link
            href="/hackathon"
            className="text-purple-900 font-normal text-sm lg:text-base hover:text-purple-700 transition-transform duration-200 hover:scale-110"
          >
            Hackathon
          </Link>
        </div>

        {/* Mobile header row */}
        <div className="flex items-center justify-between md:hidden w-full">
          <span className="font-medium text-sm text-purple-800 tracking-wide">ShiftSC AIE</span>
          <button
            type="button"
            aria-label="Toggle navigation"
            aria-expanded={isMobileMenuOpen}
            onClick={() => setIsMobileMenuOpen((open) => !open)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-gray-200 text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-white"
          >
            <span className="sr-only">Open main menu</span>
            <span className="flex flex-col gap-1.5">
              <span
                className={`block h-0.5 w-4 rounded-full bg-current transition-transform duration-200 ${
                  isMobileMenuOpen ? "translate-y-1 rotate-45" : ""
                }`}
              />
              <span
                className={`block h-0.5 w-4 rounded-full bg-current transition-opacity duration-200 ${
                  isMobileMenuOpen ? "opacity-0" : "opacity-100"
                }`}
              />
              <span
                className={`block h-0.5 w-4 rounded-full bg-current transition-transform duration-200 ${
                  isMobileMenuOpen ? "-translate-y-1 -rotate-45" : ""
                }`}
              />
            </span>
          </button>
        </div>

        {/* Mobile navigation menu */}
        {isMobileMenuOpen && (
          <div className="mt-3 flex flex-col gap-2 md:hidden text-center">
            <Link
              href="/"
              className="py-1 text-sm font-normal hover:text-purple-700"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Home
            </Link>
            <Link
              href="/work"
              className="py-1 text-sm font-normal hover:text-purple-700"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Work
            </Link>
            <Link
              href="/curriculum"
              className="py-1 text-sm font-normal hover:text-purple-700"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Curriculum
            </Link>
            <Link
              href="/team"
              className="py-1 text-sm font-normal hover:text-purple-700"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Team
            </Link>
            <Link
              href="/culture"
              className="py-1 text-sm font-normal hover:text-purple-700"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Culture
            </Link>
            <Link
              href="/hackathon"
              className="py-1 text-sm font-normal text-purple-900 hover:text-purple-700"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Hackathon
            </Link>
          </div>
        )}
      </div>
    </nav>
  );
}
