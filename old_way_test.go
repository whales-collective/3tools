package main

import (
	"3tools/client"
	"3tools/tools"
	"context"
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
)

// go test -v -run TestOldWayThenToolCalls
func TestOldWayThenToolCalls(t *testing.T) {

	jsonOutputModel := "ai/qwen2.5:latest"

	systemContentIntroduction := `You have access to the following tools:`

	catalog := tools.GetToolsCatalog()
	// make a JSON String from the content of tools
	toolsJson, err := json.Marshal(catalog)
	if err != nil {
		t.Fatalf("Error marshalling tools to JSON: %v", err)
	}
	t.Logf("Tools JSON: %s", toolsJson)
	toolsContent := "v[AVAILABLE_TOOLS]" + string(toolsJson) + "[/AVAILABLE_TOOLS]"

	systemContentInstructions := `If the question of the user matched the description of a tool, the tool will be called.
	To call a tool, respond with a JSON object with the following structure: 
	[
		{
			"name": <name of the called tool>,
			"arguments": {
				<name of the argument>: <value of the argument>
			}
		},
	]
	
	search the name of the tool in the list of tools with the Name field
	`

	content := `
		Make a Vulcan salute to Spock
		Say Hello to John Doe
		Why the sky is blue?
		Add 10 and 32
		Make a Vulcan salute to Bob Morane
		Say Hello to Jane Doe
		Who is Jean-Luc Picard?
		I'm Philippe
		Add 5 and 37
		Make a Vulcan salute to Sam
		Say hello to Alice and then make a vulcan salut to Bob
	`

	engine, err := client.GetDMRClient()
	if err != nil {
		t.Fatalf("Failed to get OpenAI client: %v", err)
	}

	params := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemContentIntroduction + "\n" + toolsContent + "\n" + systemContentInstructions),
			openai.UserMessage(content),
		},
		Model:       jsonOutputModel,
		Temperature: openai.Opt(0.0),
		//Seed:        openai.Opt(int64(42)),
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
	t.Logf("\n\n✋ First Result: %s", result)

	paramsNext := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("Return all function calls wrapped in a container object with a 'function_calls' key."),
			openai.UserMessage(result),
		},
		Model:       jsonOutputModel,
		Temperature: openai.Opt(0.0),
		//Seed:        openai.Opt(int64(42)),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{
				Type: "json_object",
			},
		},
	}

	completionNext, err := engine.Chat.Completions.New(context.Background(), paramsNext)
	if err != nil {
		t.Fatalf("Error creating chat completion for next step: %v", err)
	}
	if len(completionNext.Choices) == 0 {
		t.Fatal("No choices returned from chat completion for next step")
	}
	resultNext := completionNext.Choices[0].Message.Content
	if resultNext == "" {
		t.Fatal("No content returned from chat completion for next step")
	}
	t.Logf("\n\n🚀 Result Next: %s", resultNext)

	var commands []map[string]interface{}
	errJson := json.Unmarshal([]byte(result), &commands)
	if errJson != nil {
		t.Fatalf("Error unmarshalling JSON result: %v", errJson)
	}
	if len(commands) == 0 {
		t.Fatal("No commands found in the JSON result")
	}
	t.Logf("Commands found: %d", len(commands))
	for _, command := range commands {
		t.Logf("- Command: %v", command)
	}

}
