package main

import (
	"3tools/client"
	"3tools/tools"
	"context"
	"testing"

	"github.com/openai/openai-go"
)

func TestOnePassage(t *testing.T) {

	//model := "k33g/llama-xlam-2:8b-fc-r-q2_k"
	model := "k33g/qwen2.5:0.5b-instruct-q8_0"

	userQuestion := openai.UserMessage(`
		Make a Vulcan salute to Spock
		Say Hello to John Doe
		Add 10 and 32
		Make a Vulcan salute to Bob Morane
		Say Hello to Jane Doe
		Add 5 and 37
		Make a Vulcan salute to Sam
	`)

	engine, err := client.GetDMRClient()
	if err != nil {
		t.Fatalf("Failed to get OpenAI client: %v", err)
	}

	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			userQuestion,
		},
		Model: model,
		Tools: tools.GetToolsCatalog(),
		// Enable parallel tool calls for DMR, no need for this with Ollama
		ParallelToolCalls: openai.Bool(true),
		Temperature:       openai.Opt(0.0),
	}

	completion, err := engine.Chat.Completions.New(context.Background(), params)
	if err != nil {
		t.Fatalf("Error creating chat completion: %v", err)
	}
	if len(completion.Choices) == 0 {
		t.Fatal("No choices returned from chat completion")
	}
	toolCalls := completion.Choices[0].Message.ToolCalls
	if len(toolCalls) == 0 {
		t.Fatal("No tool calls returned from chat completion")
	}
	for _, toolCall := range toolCalls {
		t.Logf("Tool Call: %s, Arguments: %s", toolCall.Function.Name, toolCall.Function.Arguments)
	}

}
