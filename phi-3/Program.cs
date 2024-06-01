using Gradio.Net;
using System;

namespace phi3
{
    public class Program
    {
        static AIChat aiChat = new AIChat("C:\\Users\\14546\\Documents\\Data\\AI\\phi3-mini-4k-cpu");

        static async Task Main(string[] args)
        {
            App.Launch(await CreateUI());
        }

        static async Task<Blocks> CreateUI()
        {
            using (var blocks = gr.Blocks())
            {
                gr.Markdown("phi-3-mini-4k-cpu");

                var chatbot = gr.Chatbot();
                var msg = gr.MultimodalTextbox(label: "message", fileTypes: ["image"]);

                Button subBtn, clearBtn;

                using (gr.Row())
                {
                    subBtn = gr.Button("Submit");
                    clearBtn = gr.Button("Clear");
                }

                await subBtn.Click(streamingFn: input => Respond(MultimodalTextbox.Payload(input.Data[0]), Chatbot.Payload(input.Data[1])),
                    inputs: [msg, chatbot], outputs: [msg, chatbot]);

                await clearBtn.Click(fn: async input => gr.Output(null, null),
                    inputs: [msg, chatbot], outputs: [msg, chatbot]);

                return blocks;
            }
        }

        static async IAsyncEnumerable<Output> Respond(MultimodalData message, IList<ChatbotMessagePair> chatHistory)
        {
            var file = message.Files.FirstOrDefault();
            if (file != null && file.MimeType != "image/png")
                file = null;
            chatHistory.Add(new ChatbotMessagePair(message.Text, ""));
            var chat = aiChat.ChatAsync(message.Text, chatHistory.Select(x => new ChatMessage() { UserMessage = x.HumanMessage.TextMessage, AIMessage = x.AiMessage.TextMessage }).ToList());
            await foreach (var retMsg in chat)
            {
                chatHistory.Last().AiMessage.TextMessage += retMsg;
                yield return gr.Output(null, chatHistory);
            }
        }
    }
}
