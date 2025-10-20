# Chapter 8: Context & State Management

## Introduction

Unlike stateless REST APIs, conversational AI systems maintain context across multiple turns. With finite context windows, intelligent decisions about what to keep, compress, or fetch on demand become critical.

**Current context window sizes:**[1][2]
- **Claude Sonnet 4**: 1,000,000 tokens (can process entire codebases with 75,000+ lines of code)[1]
- **GPT-4 Turbo**: 128,000 tokens[2]
- **Industry standard**: Shifting from 32,000 to 128,000 tokens[3]

**The challenge:** Maintain enough context for coherent conversations without exceeding limits. LLMs have no inherent memory between calls—each API request is stateless by design.[4] Context management requires explicit strategies to maintain conversational coherence across extended interactions.

---

## The Context Window Problem

### Traditional REST: Stateless by Design

```python
# Request 1
GET /api/orders
Response: [order1, order2, order3]

# Request 2 (completely independent)
GET /api/orders/123
Response: {order details}

No shared state. No memory constraints.
```

### AI-Native: Stateful Conversations

```
Turn 1: "Show my orders"
[AI keeps: order list in context]

Turn 2: "What about the third one?"
[AI needs: context from turn 1 to know which "third one"]

Turn 3: "When will it arrive?"
[AI needs: context from turns 1 & 2]

Turn 20: "What was my first order?"
[AI needs: all 20 turns of context!]

Problem: Context window = ~200K tokens
Average turn = ~1K tokens
Maximum ~200 turns, but that's too much
```

### The Economics

```
Context Usage:
- System prompt: 5K tokens
- User profile: 1K tokens
- Tool schemas: 10K tokens
- Conversation (10 turns): 20K tokens
- Working memory: 10K tokens
Total: 46K tokens

Remaining for new content: 154K tokens

But conversations can be 50+ turns!
Must manage context intelligently.
```

---

## Context Tiers

**Memory facade pattern:** Building conversational AI applications that maintain context over time requires the "memory facade" pattern, as LLMs have no inherent memory between calls.[4][8] Modern architectures typically employ:

1. **Short-term memory**: Immediate context of current conversation, managed through mutable state[8][9]
2. **Long-term memory**: Information persisting across multiple sessions, achieved through checkpointers[9]
3. **Persistent storage**: Conversation state saved periodically, enabling recovery and rollback[10]

### The Four-Tier Model

```typescript
class ContextManager {
  // TIER 1: Persistent (Always Kept)
  persistent: {
    user_id: string;
    user_name: string;
    permissions: string[];
    preferences: {
      language: string;
      timezone: string;
      currency: string;
    };
    current_goal: string;  // "finding laptop"
  };
  
  // TIER 2: Working Memory (Recent Turns)
  workingMemory: Message[] = [
    // Last 5-10 conversation turns
    // With recent tool call results
  ];
  
  // TIER 3: Compressed History (Old Turns Summarized)
  compressedHistory: string = `
    Previous conversation summary:
    User asked about Q2 revenue (discussed growth metrics).
    Approved two transactions.
    Expressed interest in competitor analysis.
  `;
  
  // TIER 4: Reference Data (Fetch on Demand)
  referenceIds: {
    order_history: "call:get_orders_when_needed",
    product_catalog: "call:search_products_when_needed",
    documents: "call:get_document_when_needed"
  };
}
```

### Token Allocation

```
Total budget: 200K tokens

Distribution:
├─ System & Tools: 15K (7.5%)
├─ Persistent Context: 5K (2.5%)
├─ Working Memory: 30K (15%)
├─ Compressed History: 10K (5%)
├─ Response Generation: 20K (10%)
└─ Available Buffer: 120K (60%)

The 60% buffer allows for:
- Large tool responses
- Document content
- Complex queries
```

---

## Tier 1: Persistent Context

### What Always Stays

```typescript
interface PersistentContext {
  // User identity & permissions
  user_id: string;
  user_email: string;
  user_role: "user" | "admin" | "power_user";
  permissions: string[];
  
  // User preferences
  preferences: {
    language: string;
    timezone: string;
    currency: string;
    theme: "light" | "dark";
  };
  
  // Current session state
  conversation_id: string;
  started_at: Date;
  current_topic: string;
  current_goal: string;
  
  // Pending actions
  pending_approvals: string[];
  items_in_cart: string[];
  
  // Critical flags
  is_first_time_user: boolean;
  requires_onboarding: boolean;
}
```

### Why These Never Leave

```typescript
// User identity: Required for every action
if (!context.user_id) {
  throw new Error("Cannot execute without user context");
}

// Preferences: Affects every response
const response = formatResponse(data, context.preferences.language);

// Current goal: Guides AI decisions
if (context.current_goal === "purchasing") {
  // Prioritize product/checkout tools
}

// Pending actions: Critical state
if (context.pending_approvals.length > 0) {
  // Mention pending approvals in relevant responses
}
```

---

## Tier 2: Working Memory

### The Sliding Window

```typescript
class WorkingMemory {
  private messages: Message[] = [];
  private readonly MAX_MESSAGES = 10;
  private readonly MAX_TOKENS = 30000;
  
  add(message: Message) {
    this.messages.push(message);
    
    // Enforce message limit
    if (this.messages.length > this.MAX_MESSAGES) {
      this.messages.shift();  // Remove oldest
    }
    
    // Enforce token limit
    while (this.estimateTokens() > this.MAX_TOKENS) {
      this.messages.shift();
    }
  }
  
  get(): Message[] {
    return this.messages;
  }
  
  private estimateTokens(): number {
    return this.messages.reduce((sum, msg) => 
      sum + estimateTokenCount(msg.content), 0
    );
  }
}
```

### What Goes in Working Memory

```typescript
interface WorkingMemoryMessage {
  // The conversation turn
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  
  // Tool calls made
  tool_calls?: {
    tool: string;
    parameters: Record<string, any>;
    result_summary: string;  // NOT full result!
  }[];
  
  // Metadata
  turn_number: number;
  intent: string;
}

// Example
workingMemory.add({
  role: "user",
  content: "Show my recent orders",
  timestamp: new Date(),
  turn_number: 5,
  intent: "view_orders"
});

workingMemory.add({
  role: "assistant",
  content: "Here are your 5 most recent orders...",
  timestamp: new Date(),
  tool_calls: [{
    tool: "get_orders",
    parameters: { user_id: "123", limit: 5 },
    result_summary: "5 orders, total $458.32"  // Summary, not full data!
  }],
  turn_number: 5,
  intent: "view_orders"
});
```

### Smart Summarization in Working Memory

```typescript
class SmartWorkingMemory {
  add(message: Message) {
    // Summarize tool results before storing
    if (message.tool_calls) {
      message.tool_calls = message.tool_calls.map(call => ({
        ...call,
        result_summary: this.summarizeResult(call.result),
        full_result: null  // Don't keep full result
      }));
    }
    
    this.messages.push(message);
  }
  
  summarizeResult(result: any): string {
    // For arrays
    if (Array.isArray(result)) {
      return `${result.length} items`;
    }
    
    // For objects with common patterns
    if (result.orders) {
      return `${result.orders.length} orders, total $${result.total}`;
    }
    
    if (result.products) {
      return `${result.products.length} products found`;
    }
    
    // Generic summary
    return `Result: ${JSON.stringify(result).slice(0, 100)}...`;
  }
}
```

---

## Tier 3: Compressed History

**Why compression matters:** Compression saves token usage, maintains continuity (older information stays accessible), and enables faster processing through smaller input sizes.[5] Advanced compression methods can achieve compression rates of up to 32x on text reconstruction tasks while maintaining semantic fidelity.[6][7]

**Semantic compression techniques:**[7]
- Identify and preserve information-dense segments
- Remove repetitive portions
- Maintain 6:1 compression ratios on average
- Each summary sentence encodes roughly 6x the semantic density of typical prose

### When to Compress

```typescript
class ConversationCompressor {
  async compressIfNeeded(
    workingMemory: Message[],
    compressedHistory: string
  ): Promise<string> {
    const totalTokens = this.estimateTokens(workingMemory, compressedHistory);
    
    // Threshold: If approaching 80% of budget
    if (totalTokens > 0.8 * MAX_CONTEXT_TOKENS) {
      return await this.compress(workingMemory, compressedHistory);
    }
    
    return compressedHistory;
  }
  
  async compress(
    workingMemory: Message[],
    existingCompressed: string
  ): Promise<string> {
    // Take oldest 5 messages from working memory
    const toCompress = workingMemory.slice(0, 5);
    
    // Generate summary
    const newSummary = await this.generateSummary(toCompress);
    
    // Append to existing compressed history
    return existingCompressed 
      ? `${existingCompressed}\n\n${newSummary}`
      : newSummary;
  }
  
  async generateSummary(messages: Message[]): string {
    const prompt = `
      Summarize this conversation section, preserving:
      - Key facts mentioned
      - Decisions made
      - Important context for future turns
      - User preferences revealed
      
      Be concise but complete.
      
      Messages:
      ${messages.map(m => `${m.role}: ${m.content}`).join('\n')}
    `;
    
    return await llm.generate(prompt);
  }
}
```

### Example Compression

**Original (5 turns, ~5K tokens):**
```
User: "Show me laptops under $1000"
AI: "Here are 12 laptops... [detailed results]"
User: "Which has best battery?"
AI: "The Dell XPS 13 has 12-hour battery... [details]"
User: "I'll think about it"
AI: "Take your time. Let me know if you need anything."
```

**Compressed (~200 tokens):**
```
User browsed laptops under $1000, interested in battery life.
Showed Dell XPS 13 (12hr battery, $899).
User indicated interest but wanted time to decide.
No purchase made yet.
```

---

## Tier 4: Reference Data (Fetch on Demand)

### Don't Keep Everything in Context

```typescript
// ❌ BAD: Load all data into context
@app.mcp_tool()
async function get_customer_data(customer_id: string) {
  return {
    profile: {...},           // 1K tokens
    order_history: [...],     // 50K tokens!
    support_tickets: [...],   // 20K tokens!
    payment_methods: [...],   // 5K tokens
    preferences: {...}        // 2K tokens
    // Total: 78K tokens just for one customer!
  };
}

// ✅ GOOD: Return summary with fetch-on-demand IDs
@app.mcp_tool()
async function get_customer_summary(customer_id: string) {
  return {
    profile: {...},           // 1K tokens
    summary: {
      total_orders: 45,
      total_spent: 12500,
      support_tickets_open: 0,
      member_since: "2023-01-15"
    },
    available_details: {
      order_history: "get_customer_orders()",
      support_tickets: "get_customer_tickets()",
      payment_methods: "get_payment_methods()"
    }
    // Total: ~1.5K tokens
  };
}
```

### Reference by ID Pattern

```typescript
// Store references, not full data
interface OrderReference {
  order_id: string;
  order_number: string;
  date: Date;
  total: number;
  status: string;
  // NO items, NO shipping details, NO full data
}

// When user asks about specific order, fetch full data
@app.mcp_tool()
async function get_order_details(order_id: string) {
  return db.get_full_order(order_id);
}

// Context only keeps:
const context = {
  recent_orders: [
    { id: "ord_123", number: "ORD-001", total: 99.99 },
    { id: "ord_124", number: "ORD-002", total: 149.99 }
  ]
};

// Not:
const context = {
  recent_orders: [
    { 
      id: "ord_123",
      items: [/* 50 items */],
      shipping: {/* full address */},
      tracking: {/* tracking history */}
      // ... everything
    }
  ]
};
```

---

## Context Cleanup Strategies

### 1. Automatic Pruning

```typescript
class ContextPruner {
  prune(context: ConversationContext): ConversationContext {
    // Remove tool call results older than 5 turns
    context.workingMemory = context.workingMemory.map(msg => {
      if (msg.turn_number < context.current_turn - 5) {
        msg.tool_calls = msg.tool_calls?.map(call => ({
          ...call,
          result: undefined,  // Remove full result
          result_summary: call.result_summary  // Keep summary
        }));
      }
      return msg;
    });
    
    // Remove irrelevant tangents
    context.workingMemory = context.workingMemory.filter(msg =>
      this.isRelevantToCurrentGoal(msg, context.current_goal)
    );
    
    return context;
  }
  
  isRelevantToCurrentGoal(message: Message, goal: string): boolean {
    // If goal changed, older messages may be irrelevant
    if (goal === "purchasing" && message.intent === "browsing") {
      // Browsing phase complete, can remove
      return false;
    }
    
    return true;
  }
}
```

### 2. Semantic Deduplication

```typescript
class SemanticDeduplicator {
  deduplicate(messages: Message[]): Message[] {
    const unique: Message[] = [];
    
    for (const msg of messages) {
      // Check if semantically similar message exists
      const isDuplicate = unique.some(existing =>
        this.isSemanticallyS imilar(msg, existing)
      );
      
      if (!isDuplicate) {
        unique.push(msg);
      }
    }
    
    return unique;
  }
  
  isSemanticallyDuplicate(msg1: Message, msg2: Message): boolean {
    // User asked same thing multiple ways
    if (
      msg1.intent === msg2.intent &&
      this.cosineSimilarity(msg1.embedding, msg2.embedding) > 0.95
    ) {
      return true;
    }
    
    return false;
  }
}
```

### 3. Importance Scoring

```typescript
interface ImportanceScore {
  message: Message;
  score: number;
}

class ImportanceScorer {
  score(message: Message, context: ConversationContext): number {
    let score = 0;
    
    // Recent messages more important
    const age = context.current_turn - message.turn_number;
    score += Math.max(0, 10 - age);
    
    // Messages with decisions/actions important
    if (message.tool_calls && message.tool_calls.length > 0) {
      score += 5;
    }
    
    // User preferences revealed
    if (this.revealsPreference(message)) {
      score += 8;
    }
    
    // Related to current goal
    if (message.intent === context.current_goal) {
      score += 7;
    }
    
    return score;
  }
  
  keepTopN(messages: Message[], n: number, context: ConversationContext): Message[] {
    const scored = messages.map(msg => ({
      message: msg,
      score: this.score(msg, context)
    }));
    
    scored.sort((a, b) => b.score - a.score);
    
    return scored.slice(0, n).map(s => s.message);
  }
}
```

---

## Handling Long Documents

### Problem: User Uploads 100-Page PDF

```typescript
// Can't put entire document in context
const document = await readPDF("100_pages.pdf");
// document.content = 200K tokens  // Exceeds entire budget!

// Solution: Chunk and retrieve relevant sections
```

**Retrieval-Augmented Generation (RAG):** RAG improves model responses by injecting external context at runtime. Instead of relying solely on pre-trained knowledge, RAG retrieves relevant information from data sources and uses it to generate context-aware responses.[11][12]

**How RAG works:**[11][12]
1. Break knowledge base into smaller chunks (typically 256-1000 tokens)
2. Convert chunks into vector embeddings that encode semantic meaning
3. Store embeddings in vector database for semantic similarity search
4. When user asks a question, retrieve most semantically similar chunks
5. Include retrieved chunks as context in the prompt

### Chunking Strategy

**Chunking strategies:**[13][14][15]

- **Fixed-size chunking**: Every chunk kept to uniform length (e.g., 256-1000 tokens). Simple to implement, ensures consistent embedding sizes.[13][14]
- **Context-aware chunking**: Splits documents based on semantic markers like punctuation, paragraph breaks, markdown/HTML tags.[14]
- **Semantic chunking**: Segments text into meaningful, conceptually distinct groups representing coherent ideas, unlike arbitrary separators or fixed lengths.[15]

**Optimization recommendations:**[16]
- Smaller chunks (256 tokens) improve retrieval precision
- Larger chunks provide broader context
- Use chunk overlap (10-20%) to preserve continuity
- Align chunk size with use case and context windows

```typescript
class DocumentChunker {
  chunkDocument(document: string): Chunk[] {
    const chunks: Chunk[] = [];
    const CHUNK_SIZE = 1000;  // tokens
    const OVERLAP = 200;      // tokens overlap between chunks
    
    let start = 0;
    let chunkId = 0;
    
    while (start < document.length) {
      const end = start + CHUNK_SIZE;
      const chunkText = document.slice(start, end);
      
      chunks.push({
        id: chunkId++,
        text: chunkText,
        start_pos: start,
        end_pos: end,
        embedding: await generateEmbedding(chunkText)
      });
      
      start += (CHUNK_SIZE - OVERLAP);  // Move with overlap
    }
    
    return chunks;
  }
}
```

### Semantic Retrieval

**Semantic search** finds conceptually similar content—even if exact terms don't match—using vector embeddings (numerical representations of meaning).[11][17] When a user asks a question, the system converts that question into a vector and compares it to stored vectors, retrieving the most relevant text chunks.

**Contextual Retrieval enhancement:** Anthropic's Contextual Retrieval method uses Contextual Embeddings and Contextual BM25, reducing failed retrievals by 49%, and when combined with reranking, by 67%.[17]

```typescript
@app.mcp_tool()
async function search_document(
  document_id: string,
  query: string
): Promise<RelevantChunks> {
  // Get document chunks
  const chunks = await db.getDocumentChunks(document_id);

  // Generate query embedding
  const queryEmbedding = await generateEmbedding(query);
  
  // Find most relevant chunks
  const relevantChunks = chunks
    .map(chunk => ({
      chunk,
      score: cosineSimilarity(queryEmbedding, chunk.embedding)
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);  // Top 5 chunks
  
  return {
    chunks: relevantChunks.map(r => r.chunk.text),
    document_summary: await getDocumentSummary(document_id),
    message: "Found 5 relevant sections. I can search more if needed."
  };
}
```

---

## Context Architecture: Where Does Context Live?

**Critical architectural decision:** Where should conversation context be stored and managed? This choice significantly impacts mobile data usage, security, and user experience.

### Pattern 1: Server-Side Context Management (Recommended for Production)

**Architecture:**

```
┌─────────────┐
│  Mobile/Web │
│     UI      │
└──────┬──────┘
       │ Minimal: Query + session token (1-5 KB)
       ↓
┌──────────────────────────────────────┐
│  Your Backend (Context Manager)      │
│  - Loads conversation from database  │
│  - Retrieves user context            │
│  - Manages context tiers             │
│  - Adds system prompt + tools        │
└──────┬───────────────────────────────┘
       │ Full context (100-200 KB)
       ↓
┌──────────────────┐
│  Claude/OpenAI   │
│      API         │
└──────┬───────────┘
       │ Response
       ↓
┌──────────────────────────────────────┐
│  Your Backend                        │
│  - Updates conversation in database  │
│  - Compresses if needed              │
└──────┬───────────────────────────────┘
       │ Minimal: Response only (2-10 KB)
       ↓
┌─────────────┐
│  Mobile/Web │
│     UI      │
└─────────────┘
```

**Data transfer:**[18]
- **UI ↔ Backend**: 1-10 KB per turn (mobile-friendly)
- **Backend ↔ LLM**: 100-200 KB per turn (server bandwidth, not mobile data)

**Benefits:**[18][19]
- Minimal mobile data usage
- Secure API key management (stays on server)
- Centralized context persistence (works across devices)
- Better for slow connections
- Can apply prompt caching server-side
- Enables sophisticated context strategies

**Implementation:**

```python
# Backend API endpoint
@app.post("/ai/chat")
async def handle_chat(
    query: str,
    session_id: str = Cookie()
):
    # 1. Load conversation from database
    conversation = await db.get_conversation(
        user_id=session.user_id,
        conversation_id=session.conversation_id
    )

    # 2. Build full context server-side
    context = ContextManager(
        persistent=conversation.persistent_context,
        working_memory=conversation.recent_turns[-10:],
        compressed_history=conversation.compressed_history
    )

    # 3. Send full context to LLM (server-side bandwidth)
    response = await claude.messages.create(
        system=load_system_prompt(),
        messages=context.get_messages() + [{"role": "user", "content": query}],
        tools=load_relevant_tools(context.current_goal)
    )

    # 4. Update conversation in database
    await conversation.add_turn(query, response)
    await db.save_conversation(conversation)

    # 5. Return minimal response to UI
    return {"message": response.content}  # Only 2-10 KB to mobile
```

**Use when:**
- Building mobile applications
- Users on bandwidth-constrained connections
- Need multi-device sync
- Require server-side tool execution
- Want centralized context management

### Pattern 2: Client-Side Context Management

**Architecture:**

```
┌─────────────┐
│  Mobile/Web │
│     UI      │
│  - Stores   │
│    history  │
│    (local)  │
│  - Manages  │
│    context  │
└──────┬──────┘
       │ Large: Full context (100-200 KB)
       ↓
┌──────────────────┐
│  Claude/OpenAI   │
│      API         │
└──────┬───────────┘
       │ Response (10-50 KB)
       ↓
┌─────────────┐
│  Mobile/Web │
│     UI      │
│  - Updates  │
│    local    │
│    history  │
└─────────────┘
```

**Data transfer:**
- **UI ↔ LLM**: 100-200 KB per turn (heavy mobile usage)

**Benefits:**[19]
- Simpler backend (no conversation storage)
- Offline-capable (with local models)
- Faster for single-device, desktop use
- Full privacy (data stays local)

**Drawbacks:**[18][19]
- Heavy mobile data usage
- Slow on poor connections
- API keys need proxy or exposure risk
- Conversation history in browser/app storage
- No cross-device sync
- Complex mobile app

**Use when:**
- Desktop/web-only applications
- Guaranteed good bandwidth
- Using local models (Ollama)
- Privacy-critical (keep data off servers)

### Pattern 3: Hybrid (Context Hints from Client)

**Architecture:**

```
┌─────────────┐
│  Mobile/Web │
│  - Recent   │
│    context  │
│    hints    │
└──────┬──────┘
       │ Medium: Recent turns + hints (10-50 KB)
       ↓
┌──────────────────────────────────────┐
│  Your Backend                        │
│  - Merges client hints with stored   │
│    history                            │
│  - Adds system prompt + tools        │
└──────┬───────────────────────────────┘
       │ Full context (100-200 KB)
       ↓
┌──────────────────┐
│  Claude/OpenAI   │
│      API         │
└──────────────────┘
```

**Data transfer:**
- **UI ↔ Backend**: 10-50 KB (depends on history size)
- **Backend ↔ LLM**: 100-200 KB

**Benefits:**
- Client can provide local context
- Server maintains authoritative history
- Enables client-side optimization

**Use when:**
- Need client context awareness
- Want server-side security
- Hybrid desktop/mobile app

### Recommended: Server-Side for Production

**For production AI-native applications with mobile users**, server-side context management is strongly recommended:[18][19][20]

```python
# Production pattern
class ServerSideContextManager:
    def __init__(self, user_id: str, conversation_id: str):
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def process_turn(self, query: str) -> str:
        # Load from database (server-side)
        context = await self.load_context()

        # Add compression if needed
        if context.token_count > 180000:
            context = await self.compress_old_turns(context)

        # Send to LLM (server bandwidth)
        response = await self.call_llm(context, query)

        # Update database
        await self.save_context(context, query, response)

        # Return minimal response (mobile-friendly)
        return response.content
```

**Key advantages:**
1. **Mobile-friendly**: UI sends 1-10 KB per turn vs 100-200 KB
2. **Secure**: API keys never leave server
3. **Cross-device**: Conversation syncs automatically
4. **Optimizable**: Can implement prompt caching, compression server-side
5. **Reliable**: Works on slow/unreliable mobile connections

---

## Context Persistence Across Sessions

**State persistence patterns:** Modern systems employ distributed architecture combining in-memory storage for active conversations with persistent storage for long-term retention.[10] A write-through caching pattern, where state changes are simultaneously updated in both cache and persistent storage, provides robust data durability while maintaining rapid access.[10]

**Checkpointing mechanisms:** Frameworks like LangGraph offer state persistence to maintain context over long interactions, periodically saving conversation state to enable recovery from system failures and providing rollback capabilities through state versioning.[9][10]

### Saving Context

```typescript
class ContextPersistence {
  async saveConversation(conversationId: string, context: ConversationContext) {
    await db.conversations.upsert({
      id: conversationId,
      user_id: context.user_id,
      started_at: context.started_at,
      last_active: new Date(),
      
      // Persistent context
      persistent_context: context.persistent,
      
      // Compressed full history
      full_history_compressed: await this.compressFull(context),
      
      // Working memory (last 10 turns)
      working_memory: context.workingMemory,
      
      // Current state
      current_goal: context.current_goal,
      current_topic: context.current_topic
    });
  }
  
  async loadConversation(conversationId: string): Promise<ConversationContext> {
    const saved = await db.conversations.get(conversationId);
    
    return {
      persistent: saved.persistent_context,
      workingMemory: saved.working_memory,
      compressedHistory: saved.full_history_compressed,
      current_goal: saved.current_goal,
      current_topic: saved.current_topic
    };
  }
}
```

### Resuming Conversations

```
User returns after 2 days:

Load from database:
- Persistent context (user prefs, etc.)
- Compressed history: "Previously discussed laptops under $1000, 
  interested in Dell XPS 13..."
- Current goal: "considering_purchase"

AI: "Welcome back! Are you still thinking about the Dell XPS 13 
we discussed? I can help you place an order or answer any questions."
```

---

## Monitoring Context Usage

### Dashboard Metrics

```typescript
interface ContextMetrics {
  // Overall usage
  total_tokens_used: number;
  percentage_of_limit: number;
  
  // By tier
  persistent_tokens: number;
  working_memory_tokens: number;
  compressed_history_tokens: number;
  tool_results_tokens: number;
  
  // Growth rate
  tokens_per_turn: number;
  projected_exhaustion_turn: number;
  
  // Compression stats
  times_compressed: number;
  compression_ratio: number;
  oldest_uncompressed_turn: number;
}

// Alert when approaching limits
if (metrics.percentage_of_limit > 0.8) {
  alert("Context usage at 80%, will compress soon");
}
```

---

## Summary

**Context Management Strategy:**

```
TIER 1: Persistent (5K tokens)
  → User identity, preferences, current goal
  → Never removed

TIER 2: Working Memory (30K tokens)
  → Last 10 turns
  → Tool call summaries (not full results)
  → Sliding window

TIER 3: Compressed History (10K tokens)
  → Older turns summarized
  → Key facts preserved
  → Generated by LLM

TIER 4: Reference Data (0 tokens)
  → Don't load until needed
  → Store IDs/references only
  → Fetch on demand
```

---

## Key Takeaways

✓ **Context is finite** - Must actively manage within token limits

✓ **Four-tier model** - Persistent, working, compressed, reference

✓ **Summarize tool results** - Don't keep full data in context

✓ **Fetch on demand** - Load detailed data only when needed

✓ **Compress old turns** - Generate summaries to save space

✓ **Monitor usage** - Alert before hitting limits

✓ **Persist across sessions** - Save and restore context

---

## References

[1] Anthropic. "Claude Sonnet 4 now supports 1M tokens of context." https://www.anthropic.com/news/1m-context

[2] Artificial Analysis. "Claude 4 Sonnet - Intelligence, Performance & Price Analysis." https://artificialanalysis.ai/models/claude-4-sonnet

[3] IBM Research. "Why larger LLM context windows are all the rage." https://research.ibm.com/blog/larger-context-window

[4] Google. "Introduction to Conversational Context: Session, State, and Memory - Agent Development Kit." https://google.github.io/adk-docs/sessions/

[5] Agenta. "Top techniques to Manage Context Lengths in LLMs." https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms

[6] arXiv. "Recurrent Context Compression: Efficiently Expanding the Context Window of LLM." https://arxiv.org/abs/2406.06110

[7] Factory.ai. "Compressing Context." https://factory.ai/news/compressing-context

[8] Stream. "Implementing Context-Aware AI Responses in Your Chat App." https://getstream.io/blog/ai-chat-memory/

[9] Medium (Krishankant Singhal). "Giving Your AI Agents a Memory: Persistence and State in LangGraph." https://krishankantsinghal.medium.com/giving-your-ai-agents-a-memory-persistence-and-state-in-langgraph-407eb9f541d2

[10] Nexus Flow Innovations. "Conversation State Management: Technical Solutions." https://www.nexusflowinnovations.com/blog/conversation-state-management-technical-solutions

[11] OpenAI Help Center. "Retrieval Augmented Generation (RAG) and Semantic Search for GPTs." https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts

[12] Google Cloud. "What is Retrieval-Augmented Generation (RAG)?" https://cloud.google.com/use-cases/retrieval-augmented-generation

[13] Stack Overflow Blog. "Breaking up is hard to do: Chunking in RAG applications." https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/

[14] Medium (Tahir Saeed). "Chunking and Embedding Strategies in RAG: A Guide to Optimizing Retrieval-Augmented Generation." https://medium.com/@tahir.saeed_46137/chunking-and-embedding-strategies-in-rag-a-guide-to-optimizing-retrieval-augmented-generation-7c95432423b1

[15] Intelligence Factory. "Chunking Strategies for Retrieval-Augmented Generation (RAG): A Deep Dive into SemDB's Approach." https://medium.com/intelligence-factory/chunking-strategies-for-retrieval-augmented-generation-rag-a-deep-dive-into-semdbs-approach-2e0a6eb284a1

[16] Medium (Adnan Masood, PhD). "Optimizing Chunking, Embedding, and Vectorization for Retrieval-Augmented Generation." https://medium.com/@adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7

[17] Anthropic. "Introducing Contextual Retrieval." https://www.anthropic.com/news/contextual-retrieval

[18] Towards Data Science. "Implementing ML Systems tutorial: Server-side or Client-side models?" https://towardsdatascience.com/implementing-ml-systems-tutorial-server-side-or-client-side-models-3127960f9244/

[19] Medium (Nidhika Yadav, PhD). "Client-Side Language Model Interacting with Server-Side LLM." https://medium.com/@nidhikayadav/client-side-language-model-interacting-with-server-side-llm-33a8d46e5c4a

[20] Anthropic. "Prompt caching with Claude." https://www.anthropic.com/news/prompt-caching

---

**[← Previous: Chapter 7 - Security](chapter-7-security.md) | [Next: Chapter 9 - Analytics →](chapter-9-analytics.md)**