import * as vscode from "vscode";
import {
	CancellationToken,
	LanguageModelChatRequestMessage,
	LanguageModelResponsePart2,
	Progress,
	ProvideLanguageModelChatResponseOptions,
} from "vscode";
import type { HFModelItem, ReasoningConfig } from "../types";
import type {
	ChatMessageContent,
	OpenAIChatMessage,
	OpenAIToolCall,
	ReasoningDetail,
	ReasoningSummaryDetail,
	ReasoningTextDetail,
} from "./openaiTypes";
import {
	collectToolResultText,
	convertToolsToOpenAI,
	createDataUrl,
	isImageMimeType,
	isToolResultPart,
	mapRole,
} from "../utils";
import { CommonApi } from "../commonApi";
import { logger } from "../logger";

export class OpenaiApi extends CommonApi<OpenAIChatMessage, Record<string, unknown>> {
	private static readonly TC_JSON_START_RE = /<tool_call>\s*\{/g;
	private static readonly TC_FUNC_START_RE = /<function=(\w+)>\s*/g;
	private static readonly TC_END_TAG_RE = /<\/tool_call>/;
	private static readonly TC_FUNC_CLOSE_RE = /\s*<\/function>\s*$/;
	private static readonly TC_PARAM_START_RE = /<parameter=(\w+)>\s*/g;
	private static readonly TC_PARAM_CLOSE_RE = /\s*<\/parameter>\s*$/;
	private static readonly TOOL_XML_SIGNALS = ["<tool_call>", "<function="];
	private static readonly MAX_TOOL_PREFIX_BUFFER = 32;
	private _toolHealState: "buffering" | "streaming" | "draining" = "buffering";
	private _toolHealPrefixBuffer = "";
	private _toolHealContentAccum = "";
	private _toolHealFullContentAccum = "";
	private _healedToolCallsEmitted = false;

	constructor(modelId: string) {
		super(modelId);
	}

	/**
	 * Convert VS Code chat request messages into OpenAI-compatible message objects.
	 * @param messages The VS Code chat messages to convert.
	 * @param modelConfig model configuration that may affect message conversion.
	 * @returns OpenAI-compatible messages array.
	 */
	convertMessages(
		messages: readonly LanguageModelChatRequestMessage[],
		modelConfig: { includeReasoningInRequest: boolean }
	): OpenAIChatMessage[] {
		const out: OpenAIChatMessage[] = [];
		for (const m of messages) {
			const role = mapRole(m);
			const textParts: string[] = [];
			const imageParts: vscode.LanguageModelDataPart[] = [];
			const toolCalls: OpenAIToolCall[] = [];
			const toolResults: { callId: string; content: string }[] = [];
			const reasoningParts: string[] = [];

			for (const part of m.content ?? []) {
				if (part instanceof vscode.LanguageModelTextPart) {
					textParts.push(part.value);
				} else if (part instanceof vscode.LanguageModelDataPart && isImageMimeType(part.mimeType)) {
					imageParts.push(part);
				} else if (part instanceof vscode.LanguageModelToolCallPart) {
					const id = part.callId || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
					let args = "{}";
					try {
						args = JSON.stringify(part.input ?? {});
					} catch {
						args = "{}";
					}
					toolCalls.push({ id, type: "function", function: { name: part.name, arguments: args } });
				} else if (isToolResultPart(part)) {
					const callId = (part as { callId?: string }).callId ?? "";
					const content = collectToolResultText(part as { content?: ReadonlyArray<unknown> });
					toolResults.push({ callId, content });
				} else if (part instanceof vscode.LanguageModelThinkingPart) {
					const content = Array.isArray(part.value) ? part.value.join(" ") : part.value;
					reasoningParts.push(content);
				}
			}

			const joinedText = textParts.join(" ").trim();
			const joinedThinking = reasoningParts.join(" ").trim();

			// process assistant message
			if (role === "assistant") {
				const assistantMessage: OpenAIChatMessage = {
					role: "assistant",
				};

				if (joinedText) {
					assistantMessage.content = joinedText;
				}

				if (modelConfig.includeReasoningInRequest) {
					assistantMessage.reasoning_content = joinedThinking || "Next step.";
				}

				if (toolCalls.length > 0) {
					assistantMessage.tool_calls = toolCalls;
				}

				if (assistantMessage.content || assistantMessage.reasoning_content || assistantMessage.tool_calls) {
					out.push(assistantMessage);
				}
			}

			// process tool result messages
			for (const tr of toolResults) {
				out.push({ role: "tool", tool_call_id: tr.callId, content: tr.content || "" });
			}

			// process user messages
			if (role === "user") {
				if (imageParts.length > 0) {
					// multi-modal message
					const contentArray: ChatMessageContent[] = [];

					if (joinedText) {
						contentArray.push({
							type: "text",
							text: joinedText,
						});
					}

					for (const imagePart of imageParts) {
						const dataUrl = createDataUrl(imagePart);
						contentArray.push({
							type: "image_url",
							image_url: {
								url: dataUrl,
							},
						});
					}
					out.push({ role, content: contentArray });
				} else {
					// text-only message
					if (joinedText) {
						out.push({ role, content: joinedText });
					}
				}
			}

			// process system messages
			if (role === "system" && joinedText) {
				out.push({ role, content: joinedText });
			}
		}
		return out;
	}

	prepareRequestBody(
		rb: Record<string, unknown>,
		um: HFModelItem | undefined,
		options?: ProvideLanguageModelChatResponseOptions
	): Record<string, unknown> {
		// temperature
		if (um?.temperature !== undefined && um.temperature !== null) {
			rb.temperature = um.temperature;
		}

		// top_p
		if (um?.top_p !== undefined && um.top_p !== null) {
			rb.top_p = um.top_p;
		}

		// max_tokens / max_completion_tokens (mutually exclusive)
		// max_completion_tokens takes precedence (newer OpenAI standard for reasoning models)
		if (um?.max_completion_tokens !== undefined) {
			rb.max_completion_tokens = um.max_completion_tokens;
		} else if (um?.max_tokens !== undefined) {
			rb.max_tokens = um.max_tokens;
		}

		// OpenAI reasoning configuration
		if (um?.reasoning_effort !== undefined) {
			rb.reasoning_effort = um.reasoning_effort;
		}

		// enable_thinking (non-OpenRouter only)
		const enableThinking = um?.enable_thinking;
		if (enableThinking !== undefined) {
			rb.enable_thinking = enableThinking;

			if (um?.thinking_budget !== undefined) {
				rb.thinking_budget = um.thinking_budget;
			}
		}

		// thinking (Zai provider)
		if (um?.thinking?.type !== undefined) {
			rb.thinking = {
				type: um.thinking.type,
			};
		}

		// OpenRouter reasoning configuration
		if (um?.reasoning !== undefined) {
			const reasoningConfig: ReasoningConfig = um.reasoning as ReasoningConfig;
			if (reasoningConfig.enabled !== false) {
				const reasoningObj: Record<string, unknown> = {};
				const effort = reasoningConfig.effort;
				const maxTokensReasoning = reasoningConfig.max_tokens || 2000; // Default 2000 as per docs
				if (effort && effort !== "auto") {
					reasoningObj.effort = effort;
				} else {
					// If auto or unspecified, use max_tokens (Anthropic-style fallback)
					reasoningObj.max_tokens = maxTokensReasoning;
				}
				if (reasoningConfig.exclude !== undefined) {
					reasoningObj.exclude = reasoningConfig.exclude;
				}
				rb.reasoning = reasoningObj;
			}
		}

		// stop
		if (options?.modelOptions) {
			const mo = options.modelOptions as Record<string, unknown>;
			if (typeof mo.stop === "string" || Array.isArray(mo.stop)) {
				rb.stop = mo.stop;
			}
		}

		// tools
		const toolConfig = convertToolsToOpenAI(options);
		if (toolConfig.tools) {
			rb.tools = toolConfig.tools;
		}
		if (toolConfig.tool_choice) {
			rb.tool_choice = toolConfig.tool_choice;
		}

		// Configure user-defined additional parameters
		if (um?.top_k !== undefined) {
			rb.top_k = um.top_k;
		}
		if (um?.min_p !== undefined) {
			rb.min_p = um.min_p;
		}
		if (um?.frequency_penalty !== undefined) {
			rb.frequency_penalty = um.frequency_penalty;
		}
		if (um?.presence_penalty !== undefined) {
			rb.presence_penalty = um.presence_penalty;
		}
		if (um?.repetition_penalty !== undefined) {
			rb.repetition_penalty = um.repetition_penalty;
		}

		// Process extra configuration parameters
		if (um?.extra && typeof um.extra === "object") {
			// Add all extra parameters directly to the request body
			for (const [key, value] of Object.entries(um.extra)) {
				if (value !== undefined) {
					rb[key] = value;
				}
			}
		}

		return rb;
	}

	/**
	 * Read and parse the HF Router streaming (SSE-like) response and report parts.
	 * @param responseBody The readable stream body.
	 * @param progress Progress reporter for streamed parts.
	 * @param token Cancellation token.
	 */
	async processStreamingResponse(
		responseBody: ReadableStream<Uint8Array>,
		progress: Progress<LanguageModelResponsePart2>,
		token: CancellationToken
	): Promise<void> {
		const modelId = this._modelId;
		logger.debug("openai.stream.start", { modelId });

		const reader = responseBody.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		this.resetToolHealingState();

		try {
			while (true) {
				if (token.isCancellationRequested) {
					break;
				}

				const { done, value } = await reader.read();
				if (done) {
					break;
				}

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() || "";

				for (const line of lines) {
					if (!line.startsWith("data:")) {
						continue;
					}
					const data = line.slice(5).trim();
					logger.debug("openai.stream.chunk", { modelId, data });
					if (data === "[DONE]") {
						// Do not throw on [DONE]; any incomplete/empty buffers are ignored.
						await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ false);
						continue;
					}

					try {
						const parsed = JSON.parse(data);
						await this.processDelta(parsed, progress);
					} catch (e) {
						console.error("[OpenAI Provider] Failed to parse SSE chunk:", e, "data:", data);
						logger.error("openai.stream.chunk.error", {
							modelId,
							error: e instanceof Error ? e.message : String(e),
							data,
						});
					}
				}
			}
			logger.debug("openai.stream.done", { modelId });
			if (this._toolHealState === "draining" && !this._healedToolCallsEmitted) {
				const healed = OpenaiApi.parseToolCallsFromText(this._toolHealContentAccum);
				for (const tc of healed) {
					const toolName = tc.function.name;
					if (!toolName) {
						continue;
					}
					const parsedArgs = this.parseToolArguments(tc.function.arguments);
					if (!parsedArgs) {
						continue;
					}
					progress.report(new vscode.LanguageModelToolCallPart(tc.id, toolName, parsedArgs));
				}
				if (healed.length > 0) {
					this._healedToolCallsEmitted = true;
				}
			}
			if (!this._healedToolCallsEmitted) {
				const healedFromFull = OpenaiApi.parseToolCallsFromText(this._toolHealFullContentAccum);
				for (const tc of healedFromFull) {
					const toolName = tc.function.name;
					if (!toolName) {
						continue;
					}
					const parsedArgs = this.parseToolArguments(tc.function.arguments);
					if (!parsedArgs) {
						continue;
					}
					progress.report(new vscode.LanguageModelToolCallPart(tc.id, toolName, parsedArgs));
				}
				if (healedFromFull.length > 0) {
					this._healedToolCallsEmitted = true;
				}
			}
			// Some providers omit finish_reason and/or [DONE], so finalize buffered tool calls here too.
			await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ false);
		} catch (e) {
			console.error("[OpenAI Provider] Streaming response error:", e);
			logger.error("openai.stream.error", { modelId, error: e instanceof Error ? e.message : String(e) });
			throw e;
		} finally {
			reader.releaseLock();
			// If there's an active thinking sequence, end it first
			this.reportEndThinking(progress);
		}
	}

	private async processDelta(delta: Record<string, unknown>, progress: Progress<LanguageModelResponsePart2>): Promise<boolean> {
		let emitted = false;
		const choice = (delta.choices as Record<string, unknown>[] | undefined)?.[0];
		if (!choice) {
			const usage = delta.usage as Record<string, unknown> | undefined;
			if (usage && typeof usage.prompt_tokens === "number" && typeof usage.completion_tokens === "number") {
				const usageData = {
					prompt_tokens: usage.prompt_tokens,
					completion_tokens: usage.completion_tokens,
					total_tokens: (usage.total_tokens as number | undefined) ?? (usage.prompt_tokens + usage.completion_tokens),
					cached_tokens: ((usage.prompt_tokens_details as Record<string, unknown> | undefined)?.cached_tokens as number | undefined) ?? 0,
				};
				progress.report(new vscode.LanguageModelDataPart(new TextEncoder().encode(JSON.stringify(usageData)), "token_usage"));
				return true;
			}
			return false;
		}

		const deltaObj = choice.delta as Record<string, unknown> | undefined;
		const usage = delta.usage as Record<string, unknown> | undefined;
		if (usage && typeof usage.prompt_tokens === "number" && typeof usage.completion_tokens === "number") {
			const usageData = {
				prompt_tokens: usage.prompt_tokens,
				completion_tokens: usage.completion_tokens,
				total_tokens: (usage.total_tokens as number | undefined) ?? (usage.prompt_tokens + usage.completion_tokens),
				cached_tokens: ((usage.prompt_tokens_details as Record<string, unknown> | undefined)?.cached_tokens as number | undefined) ?? 0,
			};
			progress.report(new vscode.LanguageModelDataPart(new TextEncoder().encode(JSON.stringify(usageData)), "token_usage"));
			emitted = true;
		}

		// Process thinking content first (before regular text content)
		try {
			let maybeThinking =
				(choice as Record<string, unknown> | undefined)?.thinking ??
				(deltaObj as Record<string, unknown> | undefined)?.thinking ??
				(deltaObj as Record<string, unknown> | undefined)?.reasoning ??
				(deltaObj as Record<string, unknown> | undefined)?.reasoning_content;

			// OpenRouter/Claude reasoning_details array handling (new)
			const maybeReasoningDetails =
				(deltaObj as Record<string, unknown>)?.reasoning_details ??
				(choice as Record<string, unknown>)?.reasoning_details;
			if (maybeReasoningDetails && Array.isArray(maybeReasoningDetails) && maybeReasoningDetails.length > 0) {
				// Prioritize details array over simple reasoning
				const details: Array<ReasoningDetail> = maybeReasoningDetails as Array<ReasoningDetail>;
				// Sort by index to preserve order (in case out-of-order chunks)
				const sortedDetails = details.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));

				for (const detail of sortedDetails) {
					let extractedText = "";
					if (detail.type === "reasoning.summary") {
						extractedText = (detail as ReasoningSummaryDetail).summary;
					} else if (detail.type === "reasoning.text") {
						extractedText = (detail as ReasoningTextDetail).text;
					} else if (detail.type === "reasoning.encrypted") {
						extractedText = "[REDACTED]"; // As per docs
					} else {
						extractedText = JSON.stringify(detail); // Fallback for unknown
					}

					if (extractedText) {
						this.bufferThinkingContent(extractedText, progress);
						emitted = true;
					}
				}
				maybeThinking = null; // Skip simple thinking if details present
			}

			// Fallback to simple thinking if no details
			if (maybeThinking !== undefined && maybeThinking !== null) {
				let text = "";
				// let metadata: Record<string, unknown> | undefined;
				if (maybeThinking && typeof maybeThinking === "object") {
					const mt = maybeThinking as Record<string, unknown>;
					text = typeof mt["text"] === "string" ? (mt["text"] as string) : JSON.stringify(mt);
					// metadata = mt["metadata"] ? (mt["metadata"] as Record<string, unknown>) : undefined;
				} else if (typeof maybeThinking === "string") {
					text = maybeThinking;
				}
				if (text) {
					this.bufferThinkingContent(text, progress);
					emitted = true;
				}
			}
		} catch (e) {
			console.error("[OAI Compatible Model Provider] Failed to process thinking/reasoning_details:", e);
		}

		if (deltaObj?.content) {
			const content = String(deltaObj.content);
			this._toolHealFullContentAccum += content;

			if (this._toolHealState !== "streaming") {
				this._toolHealContentAccum += content;
			}

			if (this._toolHealState === "buffering") {
				this._toolHealPrefixBuffer += content;
				const check = this._toolHealPrefixBuffer.trimStart();
				const hasExact = OpenaiApi.TOOL_XML_SIGNALS.some((s) => check.startsWith(s));
				const hasPossiblePrefix =
					check.length < OpenaiApi.MAX_TOOL_PREFIX_BUFFER && OpenaiApi.TOOL_XML_SIGNALS.some((s) => s.startsWith(check));
				if (hasExact) {
					this._toolHealState = "draining";
					return emitted;
				}
				if (hasPossiblePrefix) {
					return emitted;
				}

				this._toolHealState = "streaming";
				const flushed = this._toolHealPrefixBuffer;
				this._toolHealPrefixBuffer = "";
				const xmlResBuffered = this.processXmlThinkBlocks(flushed, progress);
				if (xmlResBuffered.emittedAny) {
					emitted = true;
					return emitted;
				}
				this.reportEndThinking(progress);
				const textResBuffered = this.processTextContent(flushed, progress);
				if (textResBuffered.emittedAny) {
					this._hasEmittedAssistantText = true;
					emitted = true;
				}
				return emitted;
			}

			if (this._toolHealState === "draining") {
				return emitted;
			}

			const recoveredToolCalls = OpenaiApi.parseToolCallsFromText(content);
			if (recoveredToolCalls.length > 0) {
				for (const tc of recoveredToolCalls) {
					const toolName = tc.function.name;
					if (!toolName) {
						continue;
					}

					const parsedArgs = this.parseToolArguments(tc.function.arguments);
					if (!parsedArgs) {
						continue;
					}

					progress.report(new vscode.LanguageModelToolCallPart(tc.id, toolName, parsedArgs));
					emitted = true;
				}
				return emitted;
			}

			// Process XML think blocks or text content (mutually exclusive)
			const xmlRes = this.processXmlThinkBlocks(content, progress);
			if (xmlRes.emittedAny) {
				emitted = true;
			} else {
				// If there's an active thinking sequence, end it first
				this.reportEndThinking(progress);

				// Only process text content if no XML think blocks were emitted
				const res = this.processTextContent(content, progress);
				if (res.emittedAny) {
					this._hasEmittedAssistantText = true;
					emitted = true;
				}
			}
		}

		if (deltaObj?.tool_calls) {
			// If there's an active thinking sequence, end it first
			this.reportEndThinking(progress);

			const toolCalls = deltaObj.tool_calls as Array<Record<string, unknown>>;

			// SSEProcessor-like: if first tool call appears after text, emit a whitespace
			// to ensure any UI buffers/linkifiers are flushed without adding visible noise.
			if (!this._emittedBeginToolCallsHint && this._hasEmittedAssistantText && toolCalls.length > 0) {
				progress.report(new vscode.LanguageModelTextPart(" "));
				this._emittedBeginToolCallsHint = true;
			}

			for (const tc of toolCalls) {
				const idx = (tc.index as number) ?? 0;
				// Ignore any further deltas for an index we've already completed
				if (this._completedToolCallIndices.has(idx)) {
					continue;
				}
				const buf = this._toolCallBuffers.get(idx) ?? { args: "" };
				if (tc.id && typeof tc.id === "string") {
					buf.id = tc.id as string;
				}
				const func = tc.function as Record<string, unknown> | undefined;
				if (func?.name && typeof func.name === "string") {
					buf.name = func.name as string;
				}
				if (typeof func?.arguments === "string") {
					buf.args += func.arguments as string;
				}
				this._toolCallBuffers.set(idx, buf);

				// Emit immediately once arguments become valid JSON to avoid perceived hanging
				await this.tryEmitBufferedToolCall(idx, progress);
			}
		}

		const finish = (choice.finish_reason as string | undefined) ?? undefined;
		if (finish === "tool_calls" && this._toolHealState === "draining" && !this._healedToolCallsEmitted) {
			const healed = OpenaiApi.parseToolCallsFromText(this._toolHealContentAccum);
			for (const tc of healed) {
				const toolName = tc.function.name;
				if (!toolName) {
					continue;
				}
				const parsedArgs = this.parseToolArguments(tc.function.arguments);
				if (!parsedArgs) {
					continue;
				}
				progress.report(new vscode.LanguageModelToolCallPart(tc.id, toolName, parsedArgs));
				emitted = true;
			}
			if (healed.length > 0) {
				this._healedToolCallsEmitted = true;
			}
		}

		if ((finish === "tool_calls" || finish === "stop") && !this._healedToolCallsEmitted) {
			const healedFromFull = OpenaiApi.parseToolCallsFromText(this._toolHealFullContentAccum);
			for (const tc of healedFromFull) {
				const toolName = tc.function.name;
				if (!toolName) {
					continue;
				}
				const parsedArgs = this.parseToolArguments(tc.function.arguments);
				if (!parsedArgs) {
					continue;
				}
				progress.report(new vscode.LanguageModelToolCallPart(tc.id, toolName, parsedArgs));
				emitted = true;
			}
			if (healedFromFull.length > 0) {
				this._healedToolCallsEmitted = true;
			}
		}
		if (finish === "tool_calls" || finish === "stop") {
			// On both 'tool_calls' and 'stop', emit any buffered calls and throw on invalid JSON
			await this.flushToolCallBuffers(progress, /*throwOnInvalid*/ true);
		}
		return emitted;
	}

	private resetToolHealingState(): void {
		this._toolHealState = "buffering";
		this._toolHealPrefixBuffer = "";
		this._toolHealContentAccum = "";
		this._toolHealFullContentAccum = "";
		this._healedToolCallsEmitted = false;
	}

	private parseToolArguments(args: string): Record<string, unknown> | null {
		try {
			const parsed = JSON.parse(args) as unknown;
			if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
				return {};
			}
			return parsed as Record<string, unknown>;
		} catch {
			return null;
		}
	}

	private static parseToolCallsFromText(content: string): OpenAIToolCall[] {
		const toolCalls: OpenAIToolCall[] = [];

		for (const m of content.matchAll(this.TC_JSON_START_RE)) {
			const braceStart = m.index + m[0].length - 1;
			let depth = 0;
			let i = braceStart;
			let inString = false;
			while (i < content.length) {
				const ch = content[i];
				if (inString) {
					if (ch === "\\" && i + 1 < content.length) {
						i += 2;
						continue;
					}
					if (ch === '"') {
						inString = false;
					}
				} else if (ch === '"') {
					inString = true;
				} else if (ch === "{") {
					depth += 1;
				} else if (ch === "}") {
					depth -= 1;
					if (depth === 0) {
						break;
					}
				}
				i += 1;
			}

			if (depth !== 0) {
				continue;
			}

			const jsonStr = content.slice(braceStart, i + 1);
			try {
				const obj = JSON.parse(jsonStr) as { name?: string; arguments?: unknown };
				const args =
					typeof obj.arguments === "string"
						? obj.arguments
						: JSON.stringify((obj.arguments as Record<string, unknown> | undefined) ?? {});
				toolCalls.push({
					id: `call_${toolCalls.length}`,
					type: "function",
					function: {
						name: obj.name ?? "",
						arguments: args,
					},
				});
			} catch {
				// Ignore malformed fragments.
			}
		}

		if (toolCalls.length > 0) {
			return toolCalls;
		}

		const funcStarts = Array.from(content.matchAll(this.TC_FUNC_START_RE));
		for (let idx = 0; idx < funcStarts.length; idx++) {
			const fm = funcStarts[idx];
			const funcName = fm[1] ?? "";
			const bodyStart = (fm.index ?? 0) + fm[0].length;
			const nextFuncStart = idx + 1 < funcStarts.length ? (funcStarts[idx + 1].index ?? content.length) : content.length;
			const tail = content.slice(bodyStart);
			const endTag = tail.match(this.TC_END_TAG_RE);
			const bodyEnd = Math.min(endTag ? bodyStart + (endTag.index ?? 0) : content.length, nextFuncStart);
			let body = content.slice(bodyStart, bodyEnd);
			body = body.replace(this.TC_FUNC_CLOSE_RE, "");

			const argsObj: Record<string, unknown> = {};
			const params = Array.from(body.matchAll(this.TC_PARAM_START_RE));
			if (params.length === 1) {
				const p = params[0];
				const name = p[1] ?? "arg";
				const value = body.slice((p.index ?? 0) + p[0].length).replace(this.TC_PARAM_CLOSE_RE, "").trim();
				argsObj[name] = value;
			} else {
				for (let pidx = 0; pidx < params.length; pidx++) {
					const p = params[pidx];
					const name = p[1] ?? `arg_${pidx}`;
					const start = (p.index ?? 0) + p[0].length;
					const end = pidx + 1 < params.length ? (params[pidx + 1].index ?? body.length) : body.length;
					const value = body.slice(start, end).replace(this.TC_PARAM_CLOSE_RE, "").trim();
					argsObj[name] = value;
				}
			}

			toolCalls.push({
				id: `call_${toolCalls.length}`,
				type: "function",
				function: {
					name: funcName,
					arguments: JSON.stringify(argsObj),
				},
			});
		}

		return toolCalls;
	}

	async *createMessage(
		model: HFModelItem,
		systemPrompt: string,
		messages: { role: string; content: string }[],
		baseUrl: string,
		apiKey: string
	): AsyncGenerator<{ type: "text"; text: string }> {
		// Combine system prompt with first user message or as separate system message
		const openaiMessages = [...messages];
		if (systemPrompt) {
			openaiMessages.unshift({ role: "system", content: systemPrompt });
		}

		let requestBody: Record<string, unknown> = {
			model: model.id,
			messages: openaiMessages,
			stream: true,
		};
		requestBody = this.prepareRequestBody(requestBody, model, undefined);

		const headers = CommonApi.prepareHeaders(apiKey, model.apiMode ?? "openai", model.headers);

		const url = `${baseUrl.replace(/\/+$/, "")}/chat/completions`;

		// Make the API request
		const response = await fetch(url, {
			method: "POST",
			headers,
			body: JSON.stringify(requestBody),
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`OpenAI API request failed: [${response.status}] ${response.statusText}\n${errorText}`);
		}

		if (!response.body) {
			throw new Error("No response body from OpenAI API");
		}

		// Process the response
		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() || "";

				for (const line of lines) {
					if (!line.startsWith("data:")) {
						continue;
					}
					const data = line.slice(5).trim();
					if (data === "[DONE]") continue;

					try {
						const parsed = JSON.parse(data);

						// OpenAI-compatible streaming response
						const choice = (parsed.choices as Record<string, unknown>[] | undefined)?.[0];
						if (!choice) continue;

						const deltaObj = choice.delta as Record<string, unknown> | undefined;
						if (deltaObj?.content) {
							const content = String(deltaObj.content);
							yield { type: "text", text: content };
						}
						// Handle finish reason
						if (choice.finish_reason) break;
					} catch (e) {
						console.error("[OpenAI Provider] Failed to parse SSE chunk:", e, "data:", data);
					}
				}
			}
		} finally {
			reader.releaseLock();
		}
	}
}
