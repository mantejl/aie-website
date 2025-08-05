"use client"
import Image from "next/image"
import { useState, useEffect, useCallback } from "react"
import useEmblaCarousel from "embla-carousel-react";
import Fade from "embla-carousel-fade";

export default function ProjectsPage() {
  // Slideshow state for health case competition
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setTimeout(() => {
      window.scrollTo(0, 0);
    }, 100);
    setIsLoaded(true);
  }, []);

  // Your specific projects and events
  const projects = [
    {
      id: 1,
      title: "AIE x Health Case Competition",
      description: ( <>
      We hosted an “EthicAI Case Competition,” a one-day event where over 75 students from 20+ majors came together to develop tech-driven MVPs 
      addressing a fictional healthcare crisis. The competition was a huge success, featuring expert judges, over $900 in prizes, 
      and solutions grounded in AI, public health, policy, environmental science, and ethics.
     
      <div className="flex flex-col gap-2 mt-4"> 
        <a href="https://docs.google.com/document/d/1SgJejC_BZlys2mYIhCeWVynELRRpm6BpGOkpTznThLk/edit?usp=sharing" target="_blank" rel="noopener noreferrer" className="text-purple-700 font-medium hover:text-purple-900">EthicAI: Transforming Healthcare Case Study</a>
      </div>
      </> ),
      images: [
        "/work/aiehealth/1.jpg",
        "/work/aiehealth/2.jpg",
        "/work/aiehealth/3.jpg",
        "/work/aiehealth/4.jpg",
        "/work/aiehealth/5.jpg",
        "/work/aiehealth/6.jpg",
        "/work/aiehealth/7.jpg",
        "/work/aiehealth/8.jpg",
        "/work/aiehealth/9.jpg",
        "/work/aiehealth/10.jpg",
        "/work/aiehealth/11.jpg",
      ],
      bgColor: "bg-purple-50",
    },
    {
      id: 2,
      title: "AI Privacy & Security Workshop",
      description: (
        <>
          In collaboration with {" "}
          <a href="https://www.joinai.la/" target="_blank" rel="noopener noreferrer" className="text-purple-700 hover:text-purple-900">AI LA</a>, we delivered a “Privacy and Security in AI” workshop to over 30 industry professionals. This technical session combined presentations with interactive, hands-on activities focused on key topics such as differential privacy, data anonymization techniques, and membership inference attacks. The accompanying Jupyter notebooks, presentation slides, and curated keyword reference sheet are available below.
          <div className="flex flex-col gap-2 mt-4">
            <a href="https://colab.research.google.com/drive/1JO98xEAH-Z68UV5SVBCLMDWaFtEkj96k" target="_blank" rel="noopener noreferrer" className="text-purple-700 font-medium hover:text-purple-900">Differential Privacy Notebook</a>
            <a href="https://colab.research.google.com/drive/1_hCjnUGTvs13IHhcJggYOGLis5hqvQHP#scrollTo=9FNal32ZkNMN" target="_blank" rel="noopener noreferrer" className="text-purple-700 font-medium hover:text-purple-900">Data Anonymization Notebook</a>
            <a href="https://colab.research.google.com/drive/1SHzzqZeOaPktpHuDLaSVG2T18PM8bYG6" target="_blank" rel="noopener noreferrer" className="text-purple-700 font-medium hover:text-purple-900">Membership Inference Attack Notebook</a>
            <a href="https://docs.google.com/document/d/1Kee9PfK1nFKqtoCibDlwwN0g2hfdsAkdbZKa1X4TjeI/edit?usp=sharing" target="_blank" rel="noopener noreferrer" className="text-purple-700 font-medium hover:text-purple-900">Keyword Reference Sheet</a>
          </div>
        </>
      ),
      images: [
        "/work/workshop/im1.jpg",
        "/work/workshop/im2.jpg",
        "/work/workshop/im3.jpg",
      ],
      bgColor: "bg-purple-50",
    },
    { 
      id: 3,
      title: "School Presentations & Educational Outreach",
      description:
        "We've conducted educational presentations across multiple schools, reaching students from middle school to high school levels. Our presentations at USC Hybrid High School, Beijing 101 Middle School, and Kory Hunter Middle School have introduced hundreds of students to AI ethics concepts, making complex topics accessible and engaging for younger audiences.",
      images: [
        "/work/school/aipres.png",
        "/work/school/jack.png",
        "/work/school/beijing_presentation.png",
      ],
      bgColor: "bg-purple-50",
    },
    {
      id: 4,
      title: "Community Conversations",
      description:
        "We presented at the AI Ethics Reading Group, a forum that gathers community members to explore the implications of artificial intelligence. We also led three presentations as part of an adult workshop series hosted by So-LA Robotics, focusing on the fundamentals of ethical AI, real-world case studies, and practical frameworks for applying responsible AI practices in everyday projects.",
      images: [
        "/work/community/team.png",
        "/work/community/after.jpg",
        "/work/community/presentation3.jpg",
      ],
      bgColor: "bg-purple-50",
    },
    {
      id: 5,
      title: "Social Media Presence",
      description:
        "We produce and share engaging social media content that sheds light on AI ethics issues through creative formats. This includes our Blast to the Past series, where we draw parallels between today’s AI trends or misconceptions and historical moments in technology, offering technical context and lessons learned. Our Mythbusters posts tackle common misunderstandings about AI by breaking down complex topics into clear, accessible explanations.",
      videos: [
        "https://www.youtube.com/embed/AGy37JvHsnM"
      ],
      bgColor: "bg-purple-50",
    },
  ]

  return (
    <div
      className={`min-h-screen transition-all duration-1000 ${
        isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
      }`}
    >
      <div className="bg-purple-700 text-white py-12 mt-8 mb-6 overflow-hidden">
        <div className="container mx-auto px-4 flex flex-col md:flex-row items-center gap-8">
          <div className="flex-1 text-center md:text-left">
            <h1 className="text-5xl font-medium mb-4 animate-fade-in-up">Our Projects</h1>
            <p className="text-2xl font-light mb-6 max-w-2xl animate-fade-in-up delay-100">
              Shaping the future of responsible AI through education, outreach, and community engagement.
            </p>
          </div>
          <div className="flex-1 flex justify-center">
            <Image
              src="/work/school_p.jpg"
              alt="Presentation"
              width={500}
              height={500}
              className="rounded-2xl shadow-xl object-cover animate-fade-in"
            />
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-12">
        <div className="space-y-16">
          {projects.map((project, index) => (
            <section key={project.id} className={`rounded-2xl p-8 ${project.bgColor} shadow-lg`}>
              <div className={`grid lg:grid-cols-2 gap-8 items-center ${index % 2 === 1 ? "lg:grid-flow-dense" : ""}`}>
                <div className={`space-y-6 ${index % 2 === 1 ? "lg:order-2" : ""}`}>
                  <h2 className={`text-3xl font-medium ${project.textColor}`}>{project.title}</h2>
                  <div className={`text-lg font-light leading-relaxed`}>
                    {project.description}
                  </div>
                </div>

                <div className={`${index % 2 === 1 ? "lg:order-1" : ""}`}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {project.title === "AIE x Health Case Competition" ? (
                        <div className="md:col-span-2">
                          <EmblaCarousel images={project.images} />
                        </div>
                      ) : (
                        project.images && (
                          <>
                            {/* Main large image */}
                            <div className="md:col-span-2">
                              <div className="relative h-[300px] w-full rounded-xl overflow-hidden shadow-md">
                                <Image
                                  src={project.images[0] || "/placeholder.svg"}
                                  alt={`${project.title} - Main`}
                                  fill
                                  className={`object-cover hover:scale-105 transition-transform duration-300`}
                                  style={{ objectPosition: "center 30%" }}
                                />
                              </div>
                            </div>
                            {/* Two smaller images */}
                            <div className="relative h-[200px] w-full rounded-xl overflow-hidden shadow-md">
                              <Image
                                src={project.images[1] || "/placeholder.svg"}
                                alt={`${project.title} - Detail 1`}
                                fill
                                className="object-cover hover:scale-105 transition-transform duration-300"
                              />
                            </div>
                            <div className="relative h-[200px] w-full rounded-xl overflow-hidden shadow-md">
                              <Image
                                src={project.images[2] || "/placeholder.svg"}
                                alt={`${project.title} - Detail 2`}
                                fill
                                className="object-cover hover:scale-105 transition-transform duration-300"
                              />
                            </div>
                          </>
                        )
                      )}
                      {project.videos && (
                        <div className="relative h-[300px] w-[450px] rounded-xl overflow-hidden shadow-md bg-black flex items-center justify-center ml-24">
                          {project.videos.map((video, idx) => (
                            <div key={idx} className="relative h-[310px] w-[450px] rounded-xl overflow-hidden shadow-md bg-black flex items-center justify-center">
                              {video.includes("youtube.com") ? (
                                <iframe
                                  src={video}
                                  title={`YouTube video ${idx + 1}`}
                                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                  allowFullScreen
                                  className="w-full h-full"
                                  style={{ minHeight: 200 }}
                                />
                              ) : (
                                <video
                                  src={video}
                                  controls
                                  className="w-full h-full object-cover"
                                  style={{ maxHeight: 200 }}
                                />
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                </div>
              </div>
            </section>
          ))}
        </div>
      </div>

      <div className="bg-purple-700 text-white py-16">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-medium mb-4">Want to Collaborate?</h2>
          <p className="text-xl mb-8 max-w-2xl mx-auto font-light">
            We're always looking for new opportunities to share our knowledge and learn from others. Reach out if you'd
            like us to present at your school or event!
          </p>
          <a href="mailto:uscshiftscaie@gmail.com">
            <button className="bg-white text-purple-700 px-8 py-3 rounded-lg font-bold hover:bg-gray-100 transition-colors">
              Get In Touch
            </button>
          </a>
        </div>
      </div>
    </div>
  )
}

export function EmblaCarousel({ images }) {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: false }, [Fade()]);
  const [prevEnabled, setPrevEnabled] = useState(false);
  const [nextEnabled, setNextEnabled] = useState(false);

  // Update button states
  const onSelect = useCallback(() => {
    if (!emblaApi) return;
    setPrevEnabled(emblaApi.canScrollPrev());
    setNextEnabled(emblaApi.canScrollNext());
  }, [emblaApi]);

  // Scroll handlers
  const scrollPrev = useCallback(() => emblaApi && emblaApi.scrollPrev(), [emblaApi]);
  const scrollNext = useCallback(() => emblaApi && emblaApi.scrollNext(), [emblaApi]);

  useEffect(() => {
    if (!emblaApi) return;
    onSelect();
    emblaApi.on("select", onSelect);
    return () => emblaApi.off("select", onSelect);
  }, [emblaApi, onSelect]);

  return (
    <div className="relative w-full">
      <div ref={emblaRef} className="overflow-hidden w-full">
        <div className="flex">
          {images.map((img, idx) => (
            <div className="flex-shrink-0 w-full" key={idx}>
              <img
                src={img}
                alt={`Slide ${idx + 1}`}
                className="h-[350px] w-full object-cover rounded-xl"
              />
            </div>
          ))}
        </div>
      </div>
      {/* Prev/Next Buttons */}
      <button
        className="absolute left-2 top-1/2 -translate-y-1/2 bg-white bg-opacity-80 rounded-full p-2 shadow hover:bg-opacity-100 transition disabled:opacity-50"
        onClick={scrollPrev}
        disabled={!prevEnabled}
        aria-label="Previous slide"
      >
        &#8592;
      </button>
      <button
        className="absolute right-2 top-1/2 -translate-y-1/2 bg-white bg-opacity-80 rounded-full p-2 shadow hover:bg-opacity-100 transition disabled:opacity-50"
        onClick={scrollNext}
        disabled={!nextEnabled}
        aria-label="Next slide"
      >
        &#8594;
      </button>
    </div>
  );
}
