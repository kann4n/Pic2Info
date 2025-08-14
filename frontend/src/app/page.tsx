"use client";

import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Shield, Eye, Search, Lock, Target, Globe } from "lucide-react";

export default function Home() {
  return (
    <div className="font-mono min-h-screen flex flex-col bg-background text-foreground relative overflow-hidden">
      {/* Animated background grid */}
      <div className="fixed inset-0 bg-[linear-gradient(rgba(0,255,0,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,0,0.03)_1px,transparent_1px)] bg-[size:50px_50px] pointer-events-none" />
      
      {/* HEADER */}
      <header className="w-full py-6 px-8 flex justify-between items-center border-b border-green-500/20 backdrop-blur-sm relative z-10">
        <motion.div 
          className="flex items-center gap-3"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Shield className="text-green-400 w-8 h-8" />
          <h1 className="text-2xl font-bold text-green-400 tracking-wider">Pic2Info</h1>
          <span className="text-xs text-green-400/70 bg-green-400/10 px-2 py-1 rounded border border-green-400/30 hidden md:block">OSINT</span>
        </motion.div>
        <Button
          asChild
          variant="default"
          className="bg-green-600/20 hover:bg-green-600/30 border border-green-400/50 text-green-300 font-mono tracking-wide"
        >
          <a
            href="https://github.com/kann4n"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2"
          >
            <Globe className="w-4 h-4" />
            GitHub
          </a>
        </Button>
      </header>

      {/* HERO SECTION */}
      <section className="flex flex-col md:flex-row items-center justify-between px-8 py-16 max-w-6xl mx-auto gap-12 relative z-10">
        <motion.div
          className="flex-1"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <Target className="text-red-400 w-6 h-6" />
            <span className="text-red-400 font-mono text-sm tracking-widest">RECONNAISSANCE ENABLED</span>
          </div>
          <h2 className="text-4xl font-extrabold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-400">
            Advanced Image Intelligence
          </h2>
          <p className="text-lg text-green-300/80 mb-6 leading-relaxed">
            Leverage cutting-edge AI for comprehensive image analysis in ethical hacking and OSINT operations. 
            Extract metadata, detect objects, recognize faces, and gather actionable intelligence from visual sources.
          </p>
          <div className="flex flex-wrap gap-4 mb-8">
            <div className="flex items-center gap-2 text-sm text-cyan-400">
              <Eye className="w-4 h-4" />
              <span>Facial Recognition</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-yellow-400">
              <Search className="w-4 h-4" />
              <span>Metadata Extraction</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-purple-400">
              <Lock className="w-4 h-4" />
              <span>Reverse Image Search</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-4">
            <Button size="lg" className="bg-gradient-to-r from-green-600 to-cyan-600 hover:from-green-500 hover:to-cyan-500 text-black font-bold tracking-wide border-0">
              <a href="/upload" className="flex items-center gap-2">
                <Target className="w-4 h-4" />
                Start Analysis
              </a>
            </Button>
            <Button size="lg" variant="outline" className="border-green-400/50 text-green-400 hover:bg-green-400/10 font-mono tracking-wide">
              <a href="#capabilities" className="flex items-center gap-2">
                <Eye className="w-4 h-4" />
                View Capabilities
              </a>
            </Button>
          </div>
        </motion.div>

        <motion.div
          className="flex-1 flex justify-center relative"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-cyan-400/20 blur-xl rounded-lg" />
            <div className="relative bg-black/40 backdrop-blur-sm border border-green-400/30 rounded-lg p-4">
              <div className="text-green-400 text-xs font-mono mb-2">OSINT_ANALYSIS.exe</div>
              <div className="bg-black/60 p-4 rounded border border-green-400/20 min-h-[200px] flex flex-col justify-center">
                <div className="text-green-300 text-sm font-mono space-y-2">
                  <div>{'>'} Initializing image analysis...</div>
                  <div className="text-cyan-400">{'>'} Extracting metadata...</div>
                  <div className="text-yellow-400">{'>'} Running facial recognition...</div>
                  <div className="text-purple-400">{'>'} Cross-referencing databases...</div>
                  <div className="text-green-400">{'>'} Analysis complete. 127 data points extracted.</div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* CAPABILITIES SECTION */}
      <section id="capabilities" className="w-full py-16 px-8 border-t border-green-500/20 relative z-10">
        <motion.div
          className="max-w-6xl mx-auto"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-400">
              Intelligence Capabilities
            </h3>
            <p className="text-green-300/70 max-w-2xl mx-auto">
              Comprehensive suite of tools for ethical reconnaissance and open-source intelligence gathering.
            </p>
          </div>

          {/* Capabilities Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="bg-black/40 border-green-400/30 hover:border-green-400/50 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <Eye className="text-cyan-400 w-6 h-6" />
                  <CardTitle className="text-green-400">Object Detection</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-green-300/80 text-sm">Advanced AI models identify and classify objects, weapons, vehicles, and persons of interest.</p>
                <div className="mt-3 text-cyan-400 font-mono text-xs">Accuracy: 94.2%</div>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30 hover:border-green-400/50 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <Search className="text-yellow-400 w-6 h-6" />
                  <CardTitle className="text-green-400">Metadata Analysis</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-green-300/80 text-sm">Extract GPS coordinates, device information, timestamps, and hidden data from image files.</p>
                <div className="mt-3 text-yellow-400 font-mono text-xs">Data Points: 50+</div>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30 hover:border-green-400/50 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <Target className="text-red-400 w-6 h-6" />
                  <CardTitle className="text-green-400">Facial Recognition</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-green-300/80 text-sm">Identify and cross-reference faces against public databases and social media platforms.</p>
                <div className="mt-3 text-red-400 font-mono text-xs">Matches: Real-time</div>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30 hover:border-green-400/50 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <Globe className="text-purple-400 w-6 h-6" />
                  <CardTitle className="text-green-400">Reverse Search</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-green-300/80 text-sm">Trace image origins across the web using advanced reverse image search algorithms.</p>
                <div className="mt-3 text-purple-400 font-mono text-xs">Sources: 500M+</div>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30 hover:border-green-400/50 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <Lock className="text-orange-400 w-6 h-6" />
                  <CardTitle className="text-green-400">OCR & Text Analysis</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-green-300/80 text-sm">Extract and analyze text from images, including handwriting and encrypted messages.</p>
                <div className="mt-3 text-orange-400 font-mono text-xs">Languages: 100+</div>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30 hover:border-green-400/50 transition-all duration-300">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-3">
                  <Shield className="text-green-400 w-6 h-6" />
                  <CardTitle className="text-green-400">Threat Assessment</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-green-300/80 text-sm">Automated threat level analysis based on detected objects, locations, and contextual data.</p>
                <div className="mt-3 text-green-400 font-mono text-xs">Risk Scoring: AI-Powered</div>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      </section>

      {/* STATS SECTION */}
      <section className="w-full py-16 px-8 border-t border-green-500/20 relative z-10">
        <motion.div
          className="max-w-4xl mx-auto text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          <h3 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-400">
            Operational Metrics
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="bg-black/40 border-green-400/30">
              <CardHeader>
                <CardTitle className="text-green-400 font-mono">Detection Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-green-300 font-bold text-3xl font-mono">94.2%</p>
                <p className="text-green-400/70 text-sm mt-2">Object & Face Recognition</p>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30">
              <CardHeader>
                <CardTitle className="text-green-400 font-mono">Images Analyzed</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-cyan-400 font-bold text-3xl font-mono">25,000+</p>
                <p className="text-green-400/70 text-sm mt-2">Successful Operations</p>
              </CardContent>
            </Card>

            <Card className="bg-black/40 border-green-400/30">
              <CardHeader>
                <CardTitle className="text-green-400 font-mono">Processing Speed</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-yellow-400 font-bold text-3xl font-mono">&lt; 3s</p>
                <p className="text-green-400/70 text-sm mt-2">Average Analysis Time</p>
              </CardContent>
            </Card>
          </div>
        </motion.div>
      </section>

      {/* FOOTER */}
      <footer className="mt-auto py-6 px-8 border-t border-green-500/20 text-center text-green-400/70 relative z-10">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Shield className="w-4 h-4" />
          <span className="font-mono text-sm">Ethical Use Only</span>
        </div>
        <p className="text-xs">
          Developed by{" "}
          <a
            href="https://github.com/kann4n"
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-400 hover:text-cyan-400 transition-colors"
          >
            @kann4n
          </a>{" "}
          for responsible OSINT operations
        </p>
      </footer>
    </div>
  );
}