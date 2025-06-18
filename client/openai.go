package client

import (
	"os"

	"github.com/joho/godotenv"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func GetDMRClient() (openai.Client, error) {
	err := godotenv.Load()
	if err != nil {
		return openai.Client{}, err
	}
	baseURL := os.Getenv("MODEL_RUNNER_BASE_URL")
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(""),
	)
	return client, nil
}
