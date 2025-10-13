package service

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// MIDASRequest represents a request to the MIDAS service.
type MIDASRequest struct {
	// The text to process
	Id            string   `json:"id"`
	Text          string   `json:"text"`
	Conversations []string `json:"conversations"`
	Count         int      `json:"count"`
}

// Validate validates the request.
func (r *MIDASRequest) Validate() error {
	if r.Text == "" {
		return fmt.Errorf("text is required")
	}
	return nil
}

// MIDASResponse represents a response from the MIDAS service.
type MIDASResponse struct {
	// The text to process
	Text string `json:"text"`
}

// readMultiLineInput reads input until an empty line is entered.
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print(">>> ")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		lines = append(lines, line)
	}

	text := strings.Join(lines, "")
	fmt.Println("Sending", text)
	return text
}

var (
	threads   = 32
	tokens    = 4096
	gpulayers = -1
	seed      = -1
)

// ProcessMIDASRequest processes a MIDAS request.
func ProcessMIDASRequest(req MIDASRequest) (MIDASResponse, error) {
	// Process the request
	// t := llama.LLama

	// return MIDASResponse{
	// 	Text: strings.ToUpper(req.Text),
	// }, nil

	text := req.Text

	// callback := func(response string, bb bool) {
	// 	fmt.Println("Received response:", response)
	// }

	// Send("generate", text, callback)

	input := `
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Environment: ipython
            Tools: brave_search, wolfram_alpha
            Cutting Knowledge Date: December 2023
            Today Date: 25 Jul 2024

            # Tool Instructions
            - Always execute python code in messages that you share.
            - When looking for real time information use relevant functions if available else fallback to brave_search



            You have access to the following functions:

            Use the function 'fetch_context' to: Get precise context on any medical entity you need more context on.
            {
                "name": "fetch_context",
                "description": "Get top matching context for a given medical entity",
                "parameters": {
                    "text": {
                        "param_type": "string",
                        "description": "Medical Entity to get context on",
                        "required": true
                    },
                    "label": {
                        "param_type": "string",
                        "description": "Probable Label of the entity (Choices: Drug, Disease, Gene, Protein, Metabolite, Pathway, Tissue, Compound)",
                        "required": true
                    }
                }
            }

            If a you choose to call a function ONLY reply in the following format:
            <{start_tag}={function_name}>{parameters}{end_tag}
            where

            start_tag => <function
            parameters => a JSON dict with the function argument name as key and function argument value as value.
            end_tag => </function>

            Here is an example,
            <function=example_function_name>{"example_name": "example_value"}</function>

            Reminder:
            - Function calls MUST follow the specified format
            - Required parameters MUST be specified
            - Only call one function at a time
            - Put the entire function call reply on one line
            - Always add your sources when using search results to answer the user query

            You are a helpful assistant<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Question: """ + $$prompt$$ + \
            """<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        `
	input = strings.Replace(input, "$$prompt$$", text, -1)

	resp := MIDASResponse{
		Text: strings.ToUpper(req.Text),
	}
	return resp, nil
}
