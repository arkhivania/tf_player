using CommandLine;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using TensorFlow;

namespace tfplayer
{
    public class Options
    {
        [Value(0)]
        public string Input { get; set; }

        [Value(1)]
        public string Output { get; set; }

        [Option('m', "model", Required = true, Default = "", HelpText = "path to model file")]
        public string Model { get; set; }

        [Option('g', "forceGPU", Required = false, Default = false, HelpText = "force GPU through TF flags")]
        public bool ForceGPU { get; set; }

        [Option('p', "inputPlaceholderName", Required = true, Default = "", HelpText = "Input placeholder name")]
        public string InputPlaceholderName { get; set; }

        [Option('r', "fetchFrom", Required = true, Default = "", HelpText = "Fetch result from")]
        public string FetchFrom { get; set; }

        [Option('v', "verbose", Required = false, Default = false, HelpText = "verbose info")]
        public bool Verbose { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Parser.Default.ParseArguments<Options>(args)
                   .WithParsed<Options>(o =>
                   {
                       Run(o);
                   });
        }


        private static void Run(Options o)
        {
            using var graph = new TFGraph();
            graph.Import(File.ReadAllBytes(o.Model));

            TFSessionOptions options = new TFSessionOptions();
            if (o.ForceGPU)
                unsafe
                {
                    byte[] GPUConfig = new byte[] { 0x32, 0x02, 0x20, 0x01 };
                    fixed (void* ptr = &GPUConfig[0])
                        options.SetConfig(new IntPtr(ptr), GPUConfig.Length);
                }

            using (var session = new TFSession(graph, options))
            {
                var runner = session.GetRunner();

                if (o.Verbose)
                    foreach (var e in graph.GetEnumerator())
                        Console.WriteLine($"OT: {e.OpType} {e.Name}");

                if (o.Input.EndsWith(".png"))
                {
                    using (var image = Image.Load(o.Input))
                    {
                        var iph = graph[o.InputPlaceholderName];

                        var slice = new float[1, image.Width, image.Height, 1];
                        WriteImageToSlice(image, slice);
                        runner.AddInput($"{o.InputPlaceholderName}:0", slice);
                        runner.Fetch($"{o.FetchFrom}:0");

                        var output = runner.Run();
                        var result = output[0];
                        var r_v = (float[][])result.GetValue(true);
                        foreach (var v in r_v[0])
                            Console.WriteLine(v.ToString(CultureInfo.InvariantCulture));
                    }
                }
            }
        }

        private static void WriteImageToSlice(Image image, float[,,,] slice)
        {
            switch (image)
            {
                case Image<Rgba32> r_image:
                    for (int y = 0; y < image.Height; ++y)
                    {
                        var line = r_image.GetPixelRowSpan(y);
                        for (int x = 0; x < image.Width; ++x)
                            slice[0, x, y, 0] = line[x].G;
                    }
                    return;
                default:
                    throw new NotImplementedException("Not implemented yet");
            }
        }
    }
}
