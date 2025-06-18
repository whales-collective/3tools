package main

import (
	"3tools/client"
	"3tools/tools"
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

type Command struct {
	Complements string `json:"complements"`
	Verb        string `json:"verb"`
}

// go test -v -run TestToolCalls
func TestToolCalls(t *testing.T) {

	//jsonOutputModel := "ai/qwen2.5:3B-F16"
	jsonOutputModel := "ai/qwen2.5:latest"
	toolModel := "ai/qwen2.5:latest"

	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"verb": map[string]any{
					"type": "string",
				},
				"complements": map[string]any{
					"type": "string",
				},
			},
			"required": []string{"verb", "complements"},
		},
	}

	schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        "search_tools",
		Description: openai.String("detect verbs and complements in the text"),
		Schema:      schema,
		Strict:      openai.Bool(true),
	}

	content := `
		Make a Vulcan salute to Spock
		Say Hello to John Doe
		Why the sky is blue?
		Add 10 and 32
		Make a Vulcan salute to Bob Morane
		Say Hello to Jane Doe
		I'm Philippe
		Add 5 and 37
		Make a Vulcan salute to Sam
	`

	engine, err := client.GetDMRClient()
	if err != nil {
		t.Fatalf("Failed to get OpenAI client: %v", err)
	}

	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(content),
			openai.UserMessage("give me the list of the verbs with their complements."),
		},
		Model:       jsonOutputModel,
		Temperature: openai.Opt(0.0),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
				JSONSchema: schemaParam,
			},
		},
	}
	completion, err := engine.Chat.Completions.New(context.Background(), params)
	if err != nil {
		t.Fatalf("Error creating chat completion: %v", err)
	}
	if len(completion.Choices) == 0 {
		t.Fatal("No choices returned from chat completion")
	}
	result := completion.Choices[0].Message.Content
	if result == "" {
		t.Fatal("No content returned from chat completion")
	}
	t.Logf("Result: %s", result)

	var commands []Command
	errJson := json.Unmarshal([]byte(result), &commands)
	if errJson != nil {
		t.Fatalf("Error unmarshalling JSON result: %v", errJson)
	}
	if len(commands) == 0 {
		t.Fatal("No commands found in the JSON result")
	}

	params = openai.ChatCompletionNewParams{
		Messages:          []openai.ChatCompletionMessageParamUnion{},
		Model:             toolModel,
		Tools:             tools.GetToolsCatalog(),
		ParallelToolCalls: openai.Bool(false),
		Temperature:       openai.Opt(0.0),
	}

	for _, command := range commands {
		t.Logf("Verb: %s, Complements: %s", command.Verb, command.Complements)
		params.Messages = append(params.Messages, openai.UserMessage(command.Verb+" "+command.Complements))
		completion, err := engine.Chat.Completions.New(context.Background(), params)
		if err != nil {
			t.Fatalf("Error creating chat completion for command '%s': %v", command.Verb, err)
		}
		if len(completion.Choices) == 0 {
			t.Fatalf("No choices returned from chat completion for command '%s'", command.Verb)
		}
		toolCalls := completion.Choices[0].Message.ToolCalls
		if len(toolCalls) == 0 {
			t.Logf("😕 No tool calls returned for command '%s'", command.Verb)
		}
		for _, toolCall := range toolCalls {
			t.Logf("🎉 Tool Call: %s, Arguments: %s", toolCall.Function.Name, toolCall.Function.Arguments)
		}
		params.Messages = []openai.ChatCompletionMessageParamUnion{}

	}
	t.Logf("Total commands found: %d", len(commands))

}
