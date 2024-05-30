using Gradio.Net;

namespace phi3
{
    public class Program
    {
        static AIChat aiChat = new AIChat("./phi3-mini-4k-cpu");

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
                var msg = gr.Textbox();

                Button subBtn, clearBtn;

                using (gr.Row())
                {
                    subBtn = gr.Button("Submit");
                    clearBtn = gr.Button("Clear");
                }

                await subBtn.Click(fn: async input => await Respond(Textbox.Payload(input.Data[0]), Chatbot.Payload(input.Data[1])),
                    inputs: [msg, chatbot], outputs: [msg, chatbot]);

                await clearBtn.Click(fn: async input => gr.Output("", null),
                    inputs: [msg, chatbot], outputs: [msg, chatbot]);

                return blocks;
            }
        }

        static async Task<Output> Respond(string message, IList<ChatbotMessagePair> chatHistory)
        {
            chatHistory.Add(new ChatbotMessagePair(message, await aiChat.ChatAsync(message, chatHistory)));
            return gr.Output("", chatHistory);
        }
    }
}
