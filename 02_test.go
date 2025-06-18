package main

import (
	"3tools/client"
	"3tools/tools"
	"context"
	"fmt"
	"testing"

	"github.com/openai/openai-go"
)

// go test -v -run TestSeveralPassages
func TestSeveralPassages(t *testing.T) {

	model := "k33g/llama-xlam-2:8b-fc-r-q2_k"
	//model := "k33g/qwen2.5:0.5b-instruct-q8_0"
	//model := "ai/qwen2.5:3B-F16"
	//model := "ai/qwen2.5:1.5B-F16"

	userQuestion := openai.UserMessage(`
		TOOL1:
		Make a Vulcan salute to Spock
		TOOL2:
		Say Hello to John Doe
		TOOL3:
		Add 10 and 32
		TOOL4:
		Make a Vulcan salute to Bob Morane
		TOOL5:
		Say Hello to Jane Doe
		TOOL6:
		Add 5 and 37
		TOOL7:
		Make a Vulcan salute to Sam
	`)

	engine, err := client.GetDMRClient()
	if err != nil {
		t.Fatalf("Failed to get OpenAI client: %v", err)
	}

	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("If the tool call has been executed, detect the next tool call. And forget the previous one."),
			userQuestion,
		},
		Model: model,
		Tools: tools.GetToolsCatalog(),
		// Enable parallel tool calls for DMR, no need for this with Ollama
		ParallelToolCalls: openai.Bool(false),
		Temperature:       openai.Opt(0.0),
	}

	for {
		fmt.Println(len(params.Messages))
		completion, err := engine.Chat.Completions.New(context.Background(), params)
		if err != nil {
			t.Fatalf("Error creating chat completion: %v", err)
		}
		if len(completion.Choices) == 0 {
			t.Fatal("No choices returned from chat completion")
		}
		toolCalls := completion.Choices[0].Message.ToolCalls
		if len(toolCalls) == 0 {
			break // Aucun nouvel appel de fonction détecté
		}

		for _, toolCall := range toolCalls {
			t.Logf("Tool Call: %s, Arguments: %s", toolCall.Function.Name, toolCall.Function.Arguments)
			params.Messages = append(params.Messages, openai.ToolMessage("Exec"+toolCall.Function.Name+" "+toolCall.Function.Arguments, toolCall.ID))
			params.Messages = append(params.Messages, openai.AssistantMessage(toolCall.Function.Name+" "+toolCall.Function.Arguments))
		}
	}

}
