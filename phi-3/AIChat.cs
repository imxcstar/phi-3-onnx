using Gradio.Net;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Text;

namespace phi3
{
    public class AIChat
    {
        private string _modelPath;

        private Model _model;
        private Tokenizer _tokenizer;

        public AIChat(string modelPath)
        {
            _modelPath = modelPath;
            _model = new Model(_modelPath);
            _tokenizer = new Tokenizer(_model);
        }

        public async Task<string> ChatAsync(string message, IList<ChatbotMessagePair> chatHistory)
        {
            var chatHistoryStr = string.Join('\n', chatHistory.Select(x => @$"<|user|>
{x.HumanMessage.TextMessage}<|end|>
<|assistant|>
{x.AiMessage.TextMessage}<|end|>
"));
            if (!string.IsNullOrWhiteSpace(chatHistoryStr))
                chatHistoryStr += "\n";
            string prompt = @$"{chatHistoryStr}<|user|>
{message}<|end|>
<|assistant|>
";
            var ret = new StringBuilder();
            var sequences = _tokenizer.Encode(prompt);

            using GeneratorParams generatorParams = new GeneratorParams(_model);
            generatorParams.SetSearchOption("max_length", 200);
            generatorParams.SetInputSequences(sequences);

            using var tokenizerStream = _tokenizer.CreateStream();
            using var generator = new Generator(_model, generatorParams);
            while (!generator.IsDone())
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();
                ret.Append(tokenizerStream.Decode(generator.GetSequence(0)[^1]));
                await Task.Delay(1);
            }

            return ret.ToString();
        }
    }
}
