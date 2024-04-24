using Microsoft.ML.OnnxRuntimeGenAI;

Console.WriteLine("Loading...");

//https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
string modelPath = "./phi-3-mini/";

using Model model = new Model(modelPath);
using Tokenizer tokenizer = new Tokenizer(model);

Console.Clear();
while (true)
{
    Console.Write(">");
    string prompt = @$"<|user|>
{Console.ReadLine()}<|end|>
<|assistant|>
";
    var sequences = tokenizer.Encode(prompt);

    using GeneratorParams generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 200);
    generatorParams.SetInputSequences(sequences);

    using var tokenizerStream = tokenizer.CreateStream();
    using var generator = new Generator(model, generatorParams);
    while (!generator.IsDone())
    {
        generator.ComputeLogits();
        generator.GenerateNextToken();
        Console.Write(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
    }

    Console.WriteLine();
}