"use client"
import Link from "next/link"
import { useEffect, useState } from "react"
import { usePathname } from "next/navigation"
export default function Navbar() {
const pathname = usePathname()
const isHomePage = pathname === "/"
const [isLoaded, setIsLoaded] = useState(false)

useEffect(() => {
  setIsLoaded(true)
}, [])

  return (
    <nav className={`sticky top-5 z-50 w-full flex justify-center mt-6 bg-transparent transition-all duration-1000 ${
      isHomePage && !isLoaded ? 'opacity-0 translate-y-10' : 'opacity-100 translate-y-0'
    }`}>
      <div className="flex items-center w-[80vw] max-w-xl bg-white rounded-2xl shadow-lg px-8 py-4">
        <div className="flex-1 flex justify-center gap-12">
          <Link href="/" className="font-normal text-m hover:text-purple-700 transition-transform duration-200 hover:scale-110">Home</Link>
          <Link href="/work" className="font-normal text-m hover:text-purple-700 transition-transform duration-200 hover:scale-110">Work</Link>
          <Link href="/curriculum" className="font-normal text-m hover:text-purple-700 transition-transform duration-200 hover:scale-110">Curriculum</Link>
          <Link href="/team" className="font-normal text-m hover:text-purple-700 transition-transform duration-200 hover:scale-110">Team</Link>
          <Link href="/culture" className="font-normal text-m hover:text-purple-700 transition-transform duration-200 hover:scale-110">Culture</Link>
        </div>
      </div>
    </nav>
  );
}
