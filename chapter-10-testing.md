# Chapter 10: Testing & Quality Assurance

## Introduction

AI-native systems require a fundamentally different approach to testing. LLMs are probabilistic, not deterministic. A test that passes 95% of the time is actually a good test, not a flaky one.[21][22] This chapter explores testing strategies for AI-orchestrated applications.

**Critical requirement:** Unlike traditional software, AI models can introduce unexpected outputs, performance drift, and bias, which means engineering teams need automated guardrails to prevent failures before they reach users.[31] If no continuous tests are conducted, organizations will be at risk of deploying AI models with biased and inaccurate answers.[30]

---

## The Testing Paradigm Shift

### Traditional: Deterministic Tests

```python
# Traditional unit test
def test_create_order():
    order = create_order(
        user_id="123",
        items=["item_a", "item_b"]
    )
    
    # These assertions ALWAYS pass or ALWAYS fail
    assert order.status == "pending"
    assert order.user_id == "123"
    assert len(order.items) == 2
    
# Result: 100% pass or 100% fail
```

### AI-Native: Probabilistic Tests

```python
# AI-native test
@pytest.mark.llm_test
async def test_order_creation_intent():
    """Test AI correctly recognizes order creation intent"""
    
    test_queries = [
        "I want to buy items A and B",
        "Add A and B to my order",
        "Purchase products A and B",
        "Get me A and B",
        "Create an order with A and B"
    ]
    
    correct = 0
    for query in test_queries:
        response = await ai_agent.process(query)
        if response.tool == "create_order" and \
           "A" in response.params and "B" in response.params:
            correct += 1
    
    accuracy = correct / len(test_queries)
    
    # Statistical assertion
    assert accuracy >= 0.95, f"Accuracy {accuracy} below 95% threshold"

# Result: Passes if ≥95% correct
```

**Statistical validity:** Common thresholds for accuracy in NLP and LLM evaluation include 0.85, 0.90, and 0.95, with higher thresholds (>0.98) required for critical operations.[22][23] Statistical significance should be verified with p-values < 0.05 when comparing model performance.[24][25]

---

## The New Testing Pyramid

### Traditional Pyramid

```
        /\
       /E2E\        100 tests
      /------\
     /  API  \      500 tests
    /--------\
   /   Unit   \     2000 tests
  /____________\
```

### AI-Native Pyramid

```
        /\
       /Scen-\       50 scenarios
      /arios \      (end-to-end journeys)
     /--------\
    / Intent  \     200 tests
   / & Tool   \     (statistical accuracy)
  /------------\
 / Deterministic\   2000 tests
/    Backend     \  (business logic)
/________________\
```

**Key difference:** New middle layer for AI-specific testing.

---

## Level 1: Deterministic Backend Tests

**These don't change. Business logic is still deterministic.**

```python
# Traditional unit tests remain important
def test_payment_processing():
    """Test payment business logic"""
    payment = process_payment(
        amount=10000,  # $100 in cents
        user_id="123"
    )
    
    assert payment.status == "completed"
    assert payment.amount == 10000
    
def test_order_validation():
    """Test order validation rules"""
    with pytest.raises(ValidationError):
        create_order(
            user_id="123",
            items=[],  # Empty cart invalid
        )
```

**Why these still matter:** AI might call these functions, but the functions themselves must work correctly.

---

## Level 2: Intent & Tool Selection Tests

### Intent Recognition Tests

```python
@pytest.mark.llm_test
class TestIntentRecognition:
    """Test AI correctly identifies user intents"""
    
    async def test_payment_intent_recognition(self):
        """Test payment-related intent recognition"""
        
        payment_queries = [
            ("Charge customer $50", "create_payment"),
            ("Bill them fifty dollars", "create_payment"),
            ("Process a payment of 50", "create_payment"),
            ("Refund Jane's payment", "create_refund"),
            ("Give Jane her money back", "create_refund"),
            ("Cancel the charge", "cancel_payment")
        ]
        
        correct = 0
        for query, expected_intent in payment_queries:
            result = await ai_agent.classify_intent(query)
            if result.intent == expected_intent:
                correct += 1
        
        accuracy = correct / len(payment_queries)
        assert accuracy >= 0.95, f"Intent accuracy {accuracy:.2%}"
    
    async def test_ambiguous_intent_handling(self):
        """Test AI asks for clarification when intent unclear"""
        
        ambiguous_queries = [
            "Show me stuff",
            "Get that thing",
            "Do something with it"
        ]
        
        clarifications = 0
        for query in ambiguous_queries:
            result = await ai_agent.process(query)
            if result.type == "clarification_needed":
                clarifications += 1
        
        # Should ask for clarification most of the time
        rate = clarifications / len(ambiguous_queries)
        assert rate >= 0.8, f"Clarification rate {rate:.2%} too low"
```

### Tool Selection Tests

```python
@pytest.mark.llm_test
class TestToolSelection:
    """Test AI selects correct tools for intents"""
    
    async def test_order_tracking_tools(self):
        """Test AI uses correct tool sequence for order tracking"""
        
        test_cases = [
            {
                "query": "Where's my order #123?",
                "expected_tools": ["get_order", "get_tracking"],
                "expected_sequence": True
            },
            {
                "query": "Track order 123",
                "expected_tools": ["get_tracking"],
                "expected_sequence": False  # Can skip get_order
            }
        ]
        
        correct = 0
        for case in test_cases:
            result = await ai_agent.process(case["query"])
            
            tools_match = set(result.tools_called) == set(case["expected_tools"])
            
            if case["expected_sequence"]:
                sequence_match = result.tools_called == case["expected_tools"]
                if tools_match and sequence_match:
                    correct += 1
            else:
                if tools_match:
                    correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.90
```

### Parameter Extraction Tests

```python
@pytest.mark.llm_test
class TestParameterExtraction:
    """Test AI correctly extracts parameters from queries"""
    
    async def test_amount_extraction(self):
        """Test AI correctly converts amounts to cents"""
        
        test_cases = [
            ("Charge $50", {"amount": 5000}),
            ("Bill them 50 dollars", {"amount": 5000}),
            ("Charge fifty bucks", {"amount": 5000}),
            ("$50.99 payment", {"amount": 5099}),
            ("Process 100.00", {"amount": 10000})
        ]
        
        correct = 0
        for query, expected_params in test_cases:
            result = await ai_agent.extract_parameters(query, "create_payment")
            if result.get("amount") == expected_params["amount"]:
                correct += 1
        
        # High accuracy required for financial data
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.98, f"Amount extraction accuracy {accuracy:.2%}"
    
    async def test_date_extraction(self):
        """Test AI correctly parses dates"""
        
        today = date.today()
        test_cases = [
            ("Orders from yesterday", today - timedelta(days=1)),
            ("Last week's orders", today - timedelta(days=7)),
            ("Orders from Jan 15", date(today.year, 1, 15))
        ]
        
        correct = 0
        for query, expected_date in test_cases:
            result = await ai_agent.extract_parameters(query, "get_orders")
            extracted_date = result.get("date")
            
            # Allow 1-day tolerance for "last week" type queries
            if abs((extracted_date - expected_date).days) <= 1:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.90
```

### Hallucination Prevention & Detection Tests

**The problem:** LLMs can hallucinate non-existent tools, invalid parameters, or incorrect parameter types when orchestrating tool calls. Unlike hallucinations in text generation, tool calling hallucinations can cause runtime failures or silent data corruption.

**Common hallucination patterns:**
- Inventing tools that don't exist in the MCP server schema
- Using parameters that aren't defined in the tool specification
- Passing incorrect data types (string instead of integer, object instead of array)
- Fabricating parameter values not present in the user query
- Combining parameters from different tool signatures

#### Testing for Hallucinations

```python
@pytest.mark.llm_test
class TestHallucinationPrevention:
    """Test AI doesn't hallucinate tools or parameters"""

    async def test_non_existent_tool_rejection(self):
        """Test AI doesn't invent tools that don't exist"""

        # These queries might tempt the AI to use non-existent tools
        test_queries = [
            "Delete all user accounts",  # Might invent delete_all_users
            "Export database to CSV",     # Might invent export_database
            "Send email to all customers" # Might invent send_bulk_email
        ]

        for query in test_queries:
            result = await ai_agent.process(query)

            # AI should either:
            # 1. Use only valid tools, or
            # 2. Return clarification/error
            if result.type == "tool_call":
                assert result.tool in get_valid_tool_names(), \
                    f"Hallucinated tool: {result.tool}"

    async def test_invalid_parameter_rejection(self):
        """Test AI doesn't add parameters not in schema"""

        # Tool schema: get_order(order_id: str)
        # AI might hallucinate additional parameters

        result = await ai_agent.process(
            "Get order 123 with full details including shipping"
        )

        if result.tool == "get_order":
            valid_params = {"order_id"}
            actual_params = set(result.params.keys())

            # AI should not add hallucinated params like:
            # include_shipping, full_details, etc.
            assert actual_params <= valid_params, \
                f"Hallucinated params: {actual_params - valid_params}"

    async def test_type_validation(self):
        """Test AI uses correct parameter types"""

        # Tool schema: create_payment(amount: int, currency: str)

        test_cases = [
            ("Charge $50 USD", {"amount": 5000, "currency": "USD"}),
            ("Bill 100 euros", {"amount": 10000, "currency": "EUR"})
        ]

        for query, expected in test_cases:
            result = await ai_agent.process(query)

            if result.tool == "create_payment":
                # Verify types match schema
                assert isinstance(result.params["amount"], int), \
                    f"Amount should be int, got {type(result.params['amount'])}"
                assert isinstance(result.params["currency"], str), \
                    f"Currency should be str, got {type(result.params['currency'])}"

    async def test_parameter_fabrication_detection(self):
        """Test AI doesn't fabricate parameter values"""

        # Query doesn't mention email, so AI shouldn't add it
        query = "Create user John Doe"

        result = await ai_agent.process(query)

        if result.tool == "create_user":
            # AI might hallucinate email from the name
            if "email" in result.params:
                # Email wasn't in query, so this is fabricated
                assert False, \
                    f"Fabricated email: {result.params['email']}"
```

#### Schema Validation Enforcement

**Critical defense:** Runtime schema validation prevents hallucinated tool calls from executing.

```python
from pydantic import BaseModel, ValidationError

class CreatePaymentParams(BaseModel):
    amount: int
    currency: str
    # Only these fields allowed - no extras

@app.mcp_tool()
async def create_payment(params: CreatePaymentParams):
    """Create payment - schema strictly enforced"""

    # Pydantic automatically:
    # 1. Rejects unknown parameters
    # 2. Validates types
    # 3. Enforces required fields

    return process_payment(
        amount=params.amount,
        currency=params.currency
    )

# When AI hallucinates parameters:
try:
    await create_payment({
        "amount": 5000,
        "currency": "USD",
        "include_receipt": True  # ❌ Hallucinated!
    })
except ValidationError as e:
    # Schema validation catches hallucination
    audit_log.record_hallucination(
        tool="create_payment",
        invalid_params=["include_receipt"],
        error=str(e)
    )
    return ErrorResponse(
        type="invalid_parameters",
        message="Tool called with invalid parameters"
    )
```

#### Strict Tool Calling Mode

Many modern LLM providers support strict mode for tool calling, which enforces JSON schema compliance:[67][68]

```python
# Enable strict mode in Claude/OpenAI API calls
response = await anthropic.messages.create(
    model="claude-sonnet-4.5",
    messages=[{"role": "user", "content": query}],
    tools=[{
        "name": "create_payment",
        "description": "Create a payment",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "integer"},
                "currency": {"type": "string"}
            },
            "required": ["amount", "currency"],
            "additionalProperties": False  # Strict: no extra fields
        }
    }],
    tool_choice={"type": "auto"},
    # Request strict schema adherence
    strict=True  # Prevents hallucinated parameters
)
```

**Benefits of strict mode:**[67][68]
- **100% schema compliance** - Model guaranteed to follow exact schema
- **No hallucinated parameters** - Cannot add fields not in schema
- **Correct types enforced** - Integer fields get integers, not strings
- **Reduced latency** - Model doesn't waste tokens on invalid outputs

**Trade-offs:**
- Not all schemas supported (complex discriminated unions may fail)
- Requires well-defined schemas upfront
- May reduce flexibility in some edge cases

#### Monitoring Hallucination Rates

```python
class HallucinationMonitor:
    def record_tool_call(
        self,
        requested_tool: str,
        requested_params: dict,
        schema_valid: bool,
        execution_success: bool
    ):
        """Track hallucination metrics"""

        if not schema_valid:
            # Hallucination detected
            self.metrics.increment(
                "hallucinated_tool_calls",
                tags={
                    "tool": requested_tool,
                    "reason": "invalid_schema"
                }
            )

        # Calculate hallucination rate per tool
        self.metrics.gauge(
            f"hallucination_rate.{requested_tool}",
            self.get_hallucination_rate(requested_tool)
        )

# Alert on high hallucination rates
@app.on_event("hourly_check")
async def check_hallucination_rates():
    for tool in get_all_tools():
        rate = hallucination_monitor.get_hallucination_rate(tool)

        if rate > 0.05:  # More than 5% hallucinations
            alert.send(
                severity="warning",
                message=f"High hallucination rate for {tool}: {rate:.1%}",
                action="Review tool description and examples"
            )
```

**Key metrics to track:**
- Hallucination rate per tool (schema validation failures / total calls)
- Most commonly hallucinated parameters
- Type mismatch frequency
- Correlation between hallucinations and query complexity

**When hallucinations increase:**[69]
- Tool descriptions may be unclear or ambiguous
- Schema might be too permissive (accepting any object)
- Training data may lack sufficient examples for that tool
- Model may need fine-tuning on enterprise-specific tools (see Chapter 11)

---

## Level 2.5: Chain-of-Thought Reasoning Tests

### Testing Intermediate Reasoning Steps

**The challenge:** Chain-of-thought (CoT) prompting generates a series of intermediate reasoning steps that significantly improve complex reasoning performance.[33][34] However, testing isn't just about the final answer—you must validate the reasoning process itself.[35]

```python
@pytest.mark.llm_test
class TestChainOfThoughtReasoning:
    """Test quality of reasoning chains, not just final outputs"""

    async def test_reasoning_chain_quality(self):
        """Test that AI shows its work correctly"""

        query = "Calculate total cost: 3 items at $19.99 each, plus 8% tax"

        response = await ai_agent.process_with_reasoning(query)

        # Validate reasoning chain exists
        assert response.reasoning_steps is not None
        assert len(response.reasoning_steps) >= 3

        # Validate reasoning steps
        steps = response.reasoning_steps

        # Step 1: Should calculate subtotal
        assert "subtotal" in steps[0].lower() or "3" in steps[0]
        assert "19.99" in steps[0]

        # Step 2: Should show subtotal calculation
        assert "59.97" in steps[1] or "60" in steps[1]

        # Step 3: Should calculate tax
        assert "tax" in steps[2].lower()
        assert "8%" in steps[2] or "0.08" in steps[2]

        # Step 4: Should show final total
        assert "64.77" in steps[3] or "64.76" in steps[3]

        # Validate final answer
        assert 64.70 <= response.final_answer <= 64.80

    async def test_information_gain_per_step(self):
        """
        Test that each reasoning step adds information.
        Based on research: "A framework that quantifies the 'information
        gain' at each reasoning step"[35]
        """

        query = "If a train leaves NYC at 2pm going 60mph, and another leaves Boston at 3pm going 80mph, and they're 200 miles apart, when do they meet?"

        response = await ai_agent.process_with_reasoning(query)

        # Evaluate information gain at each step
        for i, step in enumerate(response.reasoning_steps):
            # Each step should introduce new information
            if i > 0:
                previous_concepts = extract_concepts(response.reasoning_steps[i-1])
                current_concepts = extract_concepts(step)

                # Current step should have some new concepts
                new_concepts = current_concepts - previous_concepts
                assert len(new_concepts) > 0, f"Step {i} added no new information"

        # Validate solution is correct
        assert response.validates_as_correct()
```

### Comparing Prompt Chaining vs. Chain-of-Thought

**Key difference:**[34] Prompt chaining sequences multiple prompts to break down tasks step-by-step, while CoT prompting elicits the model's reasoning process within a single prompt.

```python
@pytest.mark.llm_test
class TestOrchestrationStrategies:
    """Compare different orchestration approaches"""

    async def test_prompt_chaining_vs_cot(self):
        """Compare explicit chaining vs. implicit CoT"""

        query = "Find the cheapest laptop under $1000 and add it to cart"

        # Approach 1: Prompt chaining (explicit control)
        chain_result = await self.test_prompt_chaining(query)

        # Approach 2: Chain-of-thought (implicit reasoning)
        cot_result = await self.test_chain_of_thought(query)

        # Both should succeed, but with different characteristics
        assert chain_result.success
        assert cot_result.success

        # Chaining: More predictable, higher control
        assert chain_result.deterministic_score > 0.9
        assert chain_result.latency < cot_result.latency  # Usually faster

        # CoT: More flexible, better reasoning
        assert cot_result.reasoning_quality > chain_result.reasoning_quality
        assert cot_result.handles_edge_cases_better

    async def test_prompt_chaining(self, query: str):
        """Explicit step-by-step orchestration via code"""

        # Step 1: Search
        search_result = await ai_agent.call_tool(
            "search_products",
            {"query": "laptop", "max_price": 1000}
        )

        # Step 2: Find cheapest
        products = search_result.products
        cheapest = min(products, key=lambda p: p.price)

        # Step 3: Add to cart
        cart_result = await ai_agent.call_tool(
            "add_to_cart",
            {"product_id": cheapest.id}
        )

        return PromptChainingResult(
            success=cart_result.success,
            deterministic_score=0.95,  # Highly deterministic
            latency=search_result.latency + cart_result.latency
        )

    async def test_chain_of_thought(self, query: str):
        """Let LLM plan and reason through steps"""

        response = await ai_agent.process_with_cot(query)

        # AI decides the steps and executes them
        return CoTResult(
            success=response.goal_achieved,
            reasoning_quality=evaluate_reasoning(response.reasoning_steps),
            handles_edge_cases_better=True,  # More adaptive
            latency=response.total_latency
        )
```

**When to use each:**[36][37]
- **Prompt chaining**: When you need deterministic, predictable behavior and high performance
- **Chain-of-Thought**: When the task requires flexible reasoning and adaptation

### Testing for CoT Failure Modes

```python
@pytest.mark.llm_test
class TestCoTFailureModes:
    """Identify and test common CoT failures"""

    async def test_reasoning_loops(self):
        """Detect when AI gets stuck in reasoning loops"""

        query = "What's the square root of -1?"

        response = await ai_agent.process_with_reasoning(query)

        # Check for circular reasoning
        reasoning_texts = [step.lower() for step in response.reasoning_steps]

        # Detect repeated concepts (potential loop)
        for i in range(len(reasoning_texts) - 2):
            similarity = compute_similarity(reasoning_texts[i], reasoning_texts[i+2])
            assert similarity < 0.8, f"Reasoning loop detected at step {i}"

    async def test_hallucinated_reasoning(self):
        """Detect when AI invents incorrect intermediate steps"""

        query = "Calculate 15% tip on $42.50"

        response = await ai_agent.process_with_reasoning(query)

        # Validate each mathematical step
        for step in response.reasoning_steps:
            numbers = extract_numbers(step)
            if len(numbers) >= 2:
                # Verify arithmetic is correct
                assert verify_arithmetic(step), f"Hallucinated calculation in: {step}"

        # Final answer should be correct
        expected = 42.50 * 0.15
        assert abs(response.final_answer - expected) < 0.01
```

---

## Level 3: Scenario Tests (Integration)

### Multi-Turn Conversation Tests

```python
@pytest.mark.scenario
class TestPurchaseScenarios:
    """Test complete purchase workflows"""
    
    async def test_complete_purchase_flow(self):
        """Test full purchase from search to confirmation"""
        
        conversation = Conversation()
        
        # Turn 1: Initial request
        r1 = await conversation.user_says("I want to buy a laptop")
        assert r1.calls_tool("search_products")
        assert "laptop" in r1.tool_params["query"].lower()
        
        # Turn 2: Refinement
        r2 = await conversation.user_says("Under $1000 with good battery")
        assert r2.calls_tool("search_products") or \
               r2.calls_tool("filter_products")
        assert r2.tool_params.get("max_price") <= 1000
        
        # Turn 3: Selection
        r3 = await conversation.user_says("I'll take the second one")
        assert r3.calls_tool("add_to_cart")
        product_id = r3.tool_params.get("product_id")
        assert product_id is not None
        
        # Turn 4: Checkout
        r4 = await conversation.user_says("Checkout with my saved card")
        assert "get_cart" in r4.tools_called
        assert "get_saved_payment_methods" in r4.tools_called
        assert "create_order" in r4.tools_called
        
        # Turn 5: Confirmation
        r5 = await conversation.user_says("Confirm")
        assert r5.calls_tool("charge_payment")
        
        # Validate conversation
        assert conversation.completed_successfully()
        assert conversation.turn_count <= 8  # Efficiency check
        assert conversation.goal_achieved
```

### Multi-Agent Orchestration Testing

**Two-level evaluation framework:**[38] Test both (1) turn-level accuracy and (2) end-to-end task completion.

```python
@pytest.mark.scenario
class TestMultiAgentOrchestration:
    """Test complete workflow from input through reasoning to response"""

    async def test_end_to_end_workflow(self):
        """
        Effective evaluation examines the agent's entire workflow,
        including the full process from initial user input through
        reasoning steps and tool interactions to the final response.[38]
        """

        query = "Find me a laptop under $1000, check if it's in stock, and add to cart"

        # Track the complete workflow
        workflow = await ai_agent.process_with_tracking(query)

        # Level 1: Turn-level evaluation
        for turn in workflow.turns:
            # Each turn should have valid reasoning
            assert turn.reasoning_quality >= 0.8

            # Each tool call should succeed
            for tool_call in turn.tool_calls:
                assert tool_call.success, f"Tool {tool_call.name} failed"
                assert tool_call.latency < 2000, f"Tool {tool_call.name} too slow"

        # Level 2: End-to-end evaluation
        assert workflow.goal_achieved, "Failed to complete purchase flow"
        assert workflow.total_turns <= 6, f"Too many turns: {workflow.total_turns}"
        assert workflow.user_satisfaction_predicted >= 0.8

    async def test_task_completion_metric(self):
        """
        Task Completion (also known as task success or goal accuracy)
        is a critical metric that measures how effectively an LLM agent
        completes a user-given task.[38]
        """

        test_tasks = [
            {
                "query": "Order the cheapest available laptop",
                "success_criteria": lambda r: r.order_created and r.product_type == "laptop"
            },
            {
                "query": "Cancel my most recent order",
                "success_criteria": lambda r: r.order_cancelled and r.confirmation_sent
            },
            {
                "query": "Track all my shipments",
                "success_criteria": lambda r: r.tracking_info_provided and len(r.shipments) > 0
            }
        ]

        completed = 0
        for task in test_tasks:
            result = await ai_agent.process(task["query"])
            if task["success_criteria"](result):
                completed += 1

        # Task completion rate should be high
        completion_rate = completed / len(test_tasks)
        assert completion_rate >= 0.90, f"Task completion: {completion_rate:.2%}"
```

### LLM-as-Judge Evaluation

**Automated quality assessment:** Use GPT-4 or Claude as a judge to evaluate conversation quality.[38][39]

```python
@pytest.mark.scenario
class TestWithLLMJudge:
    """Use LLM to evaluate conversation quality"""

    async def test_conversation_quality_with_judge(self):
        """Build simulation environment leveraging GPT-4 as synthetic user"""

        # Simulate complete conversation
        conversation = await self.simulate_user_conversation(
            goal="Buy a laptop under $1000",
            user_persona="tech-savvy student"
        )

        # Use LLM-as-judge to evaluate
        evaluation = await self.evaluate_with_llm_judge(conversation)

        # Validate evaluation scores
        assert evaluation.helpfulness >= 0.8
        assert evaluation.accuracy >= 0.9
        assert evaluation.conversation_flow >= 0.85
        assert evaluation.goal_achievement == True

    async def simulate_user_conversation(self, goal: str, user_persona: str):
        """Simulate user with GPT-4"""

        conversation = Conversation()
        max_turns = 10

        for turn in range(max_turns):
            # GPT-4 acts as user
            user_message = await gpt4_user_simulator.generate_message(
                goal=goal,
                persona=user_persona,
                conversation_history=conversation.history
            )

            # Your agent responds
            agent_response = await ai_agent.process(user_message)
            conversation.add_turn(user_message, agent_response)

            # Check if goal achieved
            if conversation.goal_achieved:
                break

        return conversation

    async def evaluate_with_llm_judge(self, conversation: Conversation):
        """Use GPT-4 to judge conversation quality"""

        evaluation_prompt = f"""
        Evaluate this AI agent conversation on the following criteria:

        Conversation:
        {conversation.to_text()}

        Rate each criterion from 0.0 to 1.0:
        1. Helpfulness: Did the agent help achieve the user's goal?
        2. Accuracy: Were the agent's responses factually correct?
        3. Conversation Flow: Was the dialogue natural and efficient?
        4. Goal Achievement: Did the user accomplish their objective?

        Respond in JSON format.
        """

        judge_response = await gpt4.generate(evaluation_prompt)
        return parse_evaluation(judge_response)

    async def test_simulated_user_scenarios(self):
        """Run multiple simulated scenarios"""

        scenarios = [
            {"goal": "Buy laptop", "persona": "price-conscious"},
            {"goal": "Buy laptop", "persona": "feature-focused"},
            {"goal": "Return item", "persona": "frustrated"},
            {"goal": "Track order", "persona": "impatient"}
        ]

        results = []
        for scenario in scenarios:
            conversation = await self.simulate_user_conversation(
                goal=scenario["goal"],
                user_persona=scenario["persona"]
            )

            evaluation = await self.evaluate_with_llm_judge(conversation)
            results.append(evaluation)

        # Aggregate results
        avg_helpfulness = sum(r.helpfulness for r in results) / len(results)
        avg_goal_achievement = sum(1 for r in results if r.goal_achievement) / len(results)

        assert avg_helpfulness >= 0.80
        assert avg_goal_achievement >= 0.85
```

### Error Recovery Tests

```python
@pytest.mark.scenario
class TestErrorRecovery:
    """Test AI handles errors gracefully"""
    
    async def test_payment_declined_recovery(self):
        """Test AI recovers from payment failure"""
        
        conversation = Conversation()
        
        # Setup: User tries to checkout
        await conversation.user_says("Checkout my cart")
        
        # Inject payment failure
        inject_error("payment_declined")
        
        response = await conversation.get_next_response()
        
        # Validate recovery
        assert response.detected_error, "AI didn't detect payment failure"
        assert "declined" in response.message.lower(), "AI didn't explain error"
        assert response.offers_alternatives, "AI didn't suggest alternatives"
        assert conversation.cart_preserved, "AI lost the cart state"
        
        # User tries alternative
        recovery = await conversation.user_says("Try my other card")
        assert recovery.calls_tool("charge_payment")
        assert recovery.uses_different_payment_method
        
    async def test_out_of_stock_recovery(self):
        """Test AI handles out-of-stock gracefully"""
        
        conversation = Conversation()
        
        await conversation.user_says("Add product X to cart")
        
        # Inject out-of-stock error
        inject_error("out_of_stock", product_id="X")
        
        response = await conversation.get_next_response()
        
        assert response.detected_error
        assert response.explains_clearly
        assert response.suggests_alternatives or \
               response.offers_notification
```

---

## Level 4: Scenario-Based Evaluation (End-to-End Journeys)

### Why Tool-Level Testing Isn't Enough

**The false security:** Individual tools can each work correctly 98% of the time, yet the system can still fail to complete user journeys.[56]

```
Payment Service Tests: ✅ 98% accuracy
Auth Service Tests: ✅ 99% accuracy
Inventory Service Tests: ✅ 97% accuracy
Product Service Tests: ✅ 98% accuracy

Conclusion: System is ready! ❌ WRONG

Reality Check:
Can the AI successfully orchestrate these services to complete:
"I want to buy a laptop with my saved card"?

This requires:
1. authenticate_user() - Auth
2. search_products("laptop") - Product
3. get_product_details(id) - Product
4. add_to_cart(id) - Cart
5. get_saved_payment_methods() - Payment
6. create_order() - Order
7. charge_payment() - Payment

7 tool calls across 5 services.
Each at 98% accuracy = 0.98^7 = 86.8% success rate
```

**What can go wrong even with high individual tool accuracy:**

```
❌ Wrong tool selected for intent
User: "Show me my orders"
AI calls: get_cart() instead of get_orders()

❌ Correct tools, wrong sequence
AI calls: charge_payment() before create_order()

❌ Missing required step
AI calls: create_order() but forgets add_to_cart()

❌ Parameters from wrong context
AI uses product_id from previous conversation

❌ Poor error recovery
Payment fails, AI abandons cart instead of offering alternatives

❌ Excessive back-and-forth
Takes 12 turns for what should be 4-turn conversation
```

### The Scenario Definition Format

**Structured test scenarios for complete customer journeys:**

```json
{
  "scenario_id": "ECOM-001",
  "name": "Complete Purchase with Saved Payment",
  "category": "e-commerce",
  "priority": "critical",

  "description": "Customer discovers product, adds to cart, and completes checkout using saved payment method",

  "prerequisites": {
    "user_authenticated": true,
    "user_has_saved_payment": true,
    "cart_is_empty": true,
    "product_in_stock": true
  },

  "conversation_flow": [
    {
      "turn": 1,
      "user_input": "I need a laptop for work",
      "expected_behavior": {
        "should_call_tools": ["search_products"],
        "should_ask_user": ["budget range", "specific requirements"],
        "should_present": "product recommendations"
      },
      "success_criteria": {
        "calls_search_tool": true,
        "asks_qualifying_questions": true,
        "presents_results": true
      }
    },
    {
      "turn": 2,
      "user_input": "Under $1500 with good battery life",
      "expected_behavior": {
        "should_call_tools": ["search_products"],
        "should_filter_by": {
          "max_price": 1500,
          "features_required": ["long_battery_life"]
        },
        "should_present": "filtered results with battery info highlighted"
      }
    },
    {
      "turn": 3,
      "user_input": "I'll take the second one",
      "expected_behavior": {
        "should_call_tools": ["get_product_details", "add_to_cart"],
        "should_confirm": "Added [Product Name] to cart",
        "should_ask": "Would you like to checkout?"
      }
    },
    {
      "turn": 4,
      "user_input": "Yes, checkout with my saved card",
      "expected_behavior": {
        "should_call_tools_in_sequence": [
          "get_cart",
          "get_saved_payment_methods",
          "create_order",
          "charge_payment"
        ],
        "must_confirm_before_charge": true,
        "should_show": {
          "order_total": true,
          "payment_method_last_4": true,
          "shipping_address": true
        }
      },
      "critical_validations": [
        "Must show total before charging",
        "Must confirm payment method with user",
        "Must handle payment failure gracefully",
        "Must not charge without explicit confirmation"
      ]
    }
  ],

  "success_metrics": {
    "conversation_completed": true,
    "all_critical_tools_called": true,
    "tools_in_correct_sequence": true,
    "no_unnecessary_clarifications": true,
    "max_turns_allowed": 8,
    "user_satisfaction_threshold": 4.0
  },

  "failure_scenarios": [
    {
      "inject_at_turn": 4,
      "failure_type": "payment_declined",
      "expected_recovery": {
        "ai_detects_failure": true,
        "ai_explains_clearly": true,
        "ai_offers_alternatives": ["try different card", "use different payment method"],
        "ai_preserves_cart": true,
        "ai_doesnt_abandon_order": true
      }
    },
    {
      "inject_at_turn": 3,
      "failure_type": "product_out_of_stock",
      "expected_recovery": {
        "ai_informs_user": true,
        "ai_suggests_similar_products": true,
        "ai_offers_notify_when_available": true
      }
    }
  ]
}
```

### The Scenario Test Runner

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio

@dataclass
class TurnResult:
    turn_number: int
    user_input: str
    ai_response: str
    tools_called: List[str]
    tool_parameters: List[Dict]
    validation_passed: bool
    validation_details: Dict
    latency_ms: float

@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    turns: List[TurnResult]
    metrics: Dict[str, Any]
    failures: List[str]

class ScenarioExecutor:
    def __init__(self, ai_agent, mcp_services):
        self.ai_agent = ai_agent
        self.mcp_services = mcp_services

    async def execute_scenario(self, scenario: Dict) -> ScenarioResult:
        """Execute a complete scenario"""

        # Setup prerequisites
        await self.setup_prerequisites(scenario["prerequisites"])

        # Create conversation context
        conversation = Conversation(scenario_id=scenario["scenario_id"])

        turn_results = []
        failures = []

        # Execute each turn
        for turn_spec in scenario["conversation_flow"]:
            turn_result = await self.execute_turn(conversation, turn_spec)
            turn_results.append(turn_result)

            # Validate turn
            if not turn_result.validation_passed:
                failures.append(
                    f"Turn {turn_spec['turn']}: {turn_result.validation_details}"
                )

        # Test failure recovery
        recovery_results = []
        for failure_scenario in scenario.get("failure_scenarios", []):
            recovery = await self.test_failure_recovery(scenario, failure_scenario)
            recovery_results.append(recovery)

            if not recovery.passed:
                failures.append(f"Recovery failed: {recovery.details}")

        # Calculate metrics
        metrics = self.calculate_metrics(
            turn_results,
            recovery_results,
            scenario["success_metrics"]
        )

        # Overall pass/fail
        passed = len(failures) == 0 and all(
            m >= threshold
            for m, threshold in metrics.items()
            if "threshold" in str(threshold)
        )

        return ScenarioResult(
            scenario_id=scenario["scenario_id"],
            passed=passed,
            turns=turn_results,
            metrics=metrics,
            failures=failures
        )

    async def execute_turn(
        self,
        conversation: Conversation,
        turn_spec: Dict
    ) -> TurnResult:
        """Execute a single conversation turn"""

        import time
        start_time = time.time()

        # Send user input
        user_input = turn_spec["user_input"]
        ai_response = await conversation.send(user_input)

        latency = (time.time() - start_time) * 1000

        # Extract what tools were called
        tools_called = [call.tool_name for call in ai_response.tool_calls]
        tool_parameters = [call.parameters for call in ai_response.tool_calls]

        # Validate against expectations
        expected = turn_spec["expected_behavior"]
        validation = self.validate_turn(
            ai_response,
            expected,
            turn_spec.get("success_criteria", {})
        )

        return TurnResult(
            turn_number=turn_spec["turn"],
            user_input=user_input,
            ai_response=ai_response.text,
            tools_called=tools_called,
            tool_parameters=tool_parameters,
            validation_passed=validation["passed"],
            validation_details=validation,
            latency_ms=latency
        )

    def validate_turn(
        self,
        ai_response: AIResponse,
        expected: Dict,
        criteria: Dict
    ) -> Dict:
        """Validate AI response against expectations"""

        validations = {}

        # Check tool calls
        if "should_call_tools" in expected:
            expected_tools = set(expected["should_call_tools"])
            actual_tools = set(ai_response.tool_names)
            validations["correct_tools"] = expected_tools == actual_tools

        # Check tool sequence
        if "should_call_tools_in_sequence" in expected:
            expected_seq = expected["should_call_tools_in_sequence"]
            actual_seq = ai_response.tool_names
            validations["correct_sequence"] = expected_seq == actual_seq

        # Check if AI asked appropriate questions
        if "should_ask_user" in expected:
            for question_topic in expected["should_ask_user"]:
                asked = question_topic.lower() in ai_response.text.lower()
                validations[f"asked_about_{question_topic}"] = asked

        # Check confirmation for critical actions
        if expected.get("must_confirm_before_charge"):
            has_confirmation = "confirm" in ai_response.text.lower()
            validations["requires_confirmation"] = has_confirmation

        # Overall pass
        validations["passed"] = all(validations.values())

        return validations

    async def test_failure_recovery(
        self,
        scenario: Dict,
        failure_spec: Dict
    ) -> Dict:
        """Test how AI handles failures"""

        # Create fresh conversation
        conversation = Conversation()

        # Execute up to failure point
        failure_turn = failure_spec["inject_at_turn"]
        for turn_spec in scenario["conversation_flow"][:failure_turn - 1]:
            await conversation.send(turn_spec["user_input"])

        # Inject failure
        self.inject_failure(failure_spec["failure_type"])

        # Continue with the turn that should fail
        turn_spec = scenario["conversation_flow"][failure_turn - 1]
        response = await conversation.send(turn_spec["user_input"])

        # Validate recovery
        expected_recovery = failure_spec["expected_recovery"]

        recovery_quality = {
            "detected_error": response.detected_error,
            "explained_clearly": self.check_explanation(response.text),
            "offered_alternatives": self.check_alternatives(response.text),
            "preserved_state": await self.check_state_preserved(conversation)
        }

        passed = all(
            recovery_quality.get(key, False)
            for key in expected_recovery.keys()
        )

        return {"passed": passed, "details": recovery_quality}

    def calculate_metrics(
        self,
        turn_results: List[TurnResult],
        recovery_results: List[Dict],
        success_metrics: Dict
    ) -> Dict:
        """Calculate scenario success metrics"""

        return {
            "turns_to_completion": len(turn_results),
            "avg_latency_ms": sum(t.latency_ms for t in turn_results) / len(turn_results),
            "tool_selection_accuracy": sum(
                1 for t in turn_results if t.validation_passed
            ) / len(turn_results),
            "error_recovery_rate": sum(
                1 for r in recovery_results if r["passed"]
            ) / max(len(recovery_results), 1),
            "conversation_efficiency": len(turn_results) / success_metrics.get("max_turns_allowed", 10)
        }
```

### Running the Evaluation Suite

```bash
# Run all scenarios
$ python run_scenarios.py --suite e-commerce --model claude-sonnet-4.5-acme

Running E-Commerce Scenario Suite
==================================

ECOM-001: Complete Purchase ........................ ✓ PASS (4.2s)
ECOM-002: Browse and Filter ........................ ✓ PASS (3.8s)
ECOM-003: Return Request ........................... ✓ PASS (5.1s)
ECOM-004: Order Tracking ........................... ✓ PASS (2.9s)
ECOM-005: Payment Failure Recovery ................. ✓ PASS (6.3s)
ECOM-006: Out of Stock Handling .................... ✓ PASS (4.7s)
ECOM-007: Multi-item Purchase ...................... ✗ FAIL (8.2s)
  - Turn 4: AI called charge_payment before create_order
  - Sequence error: critical
ECOM-008: Gift Card Purchase ....................... ✓ PASS (4.5s)
ECOM-009: Subscription Signup ...................... ✓ PASS (5.9s)
ECOM-010: Account Management ....................... ✓ PASS (3.2s)

Results:
========
Total: 10 scenarios
Passed: 9 (90%)
Failed: 1 (10%)
Avg completion time: 4.88s

Critical scenarios: 5
Critical passed: 4 (80%) ⚠️  BELOW THRESHOLD (95%)

DEPLOYMENT BLOCKED ❌
```

### The Cross-Service Testing Team

**The organizational challenge:** While individual service teams can test their own tools effectively, cross-service user journeys require a centralized team with holistic understanding of the entire system.

**Why service-level testing isn't enough:**

```
Decentralized Test Ownership:
├─ Auth Team: Tests authentication_user(), get_user_profile() ✅
├─ Product Team: Tests search_products(), get_product_details() ✅
├─ Cart Team: Tests add_to_cart(), get_cart() ✅
├─ Order Team: Tests create_order(), get_order_status() ✅
└─ Payment Team: Tests get_payment_methods(), charge_payment() ✅

All individual services pass their tests at 98%+ accuracy!

But who tests this user journey?
"I want to buy the laptop I was looking at yesterday with my saved card"

This requires:
1. authenticate_user() - Auth service
2. get_browsing_history() - Analytics service
3. search_products(from_history) - Product service
4. get_product_details(id) - Product service
5. check_inventory(id) - Inventory service
6. add_to_cart(id) - Cart service
7. get_saved_payment_methods() - Payment service
8. create_order() - Order service
9. charge_payment() - Payment service

❌ No single service team understands this entire journey
❌ No service team knows how the AI should orchestrate across services
❌ No service team can validate the conversational flow makes sense to users
```

**The solution: Centralized Cross-Service Testing Team**

```
Cross-Service Testing Team Responsibilities:
├─ Define end-to-end user scenarios spanning multiple services
├─ Understand how real users interact with the system
├─ Know which services need to be orchestrated for each user goal
├─ Test AI's ability to chain tools across service boundaries
├─ Validate conversational UX across multi-step journeys
├─ Define deployment gates for cross-service scenarios
└─ Own the scenario evaluation framework and test suites
```

**Required team composition and skills:**

```python
class CrossServiceTestingTeam:
    """
    Dedicated team responsible for testing scenarios
    that span multiple services and tools.
    """

    required_skills = {
        "system_architecture": {
            "description": "Deep understanding of all services and their tools",
            "knowledge_areas": [
                "All available MCP tools across all services",
                "Service dependencies and data flow",
                "Authentication and permission boundaries",
                "Error handling across service boundaries"
            ]
        },

        "user_journey_expertise": {
            "description": "Understanding of real user behaviors and goals",
            "knowledge_areas": [
                "Common user workflows and use cases",
                "User intent mapping to tool sequences",
                "Conversational UX best practices",
                "Failure recovery expectations"
            ]
        },

        "ai_orchestration_understanding": {
            "description": "How the AI model orchestrates across services",
            "knowledge_areas": [
                "Model's tool selection behavior",
                "Context management across turns",
                "Multi-step reasoning patterns",
                "Common orchestration failures"
            ]
        },

        "cross_functional_collaboration": {
            "description": "Work with all service teams",
            "responsibilities": [
                "Gather tool specifications from each team",
                "Report failures back to owning teams",
                "Coordinate on scenario definitions",
                "Share insights on tool usage patterns"
            ]
        }
    }

    team_structure = {
        "size": "3-5 engineers for typical enterprise system",
        "reporting": "Reports to platform/infrastructure org, not individual services",
        "autonomy": "Can block deployments based on scenario test failures",
        "meeting_cadence": {
            "weekly_sync": "With all service teams to review new tools",
            "scenario_reviews": "Monthly review of scenario coverage",
            "incident_reviews": "Immediate review when production scenarios fail"
        }
    }
```

**Workflow: How the cross-service team operates**

```python
class CrossServiceTestWorkflow:
    """End-to-end workflow for cross-service testing"""

    async def define_scenario(self, user_goal: str):
        """
        Team analyzes user goal and defines expected journey
        """

        # Step 1: Understand what services are involved
        services_needed = self.analyze_required_services(user_goal)
        # Example: "Buy laptop" → [Auth, Product, Cart, Order, Payment]

        # Step 2: Map to specific tools
        tool_sequence = self.map_to_tool_calls(user_goal, services_needed)
        # Example: [authenticate_user, search_products, get_details,
        #          add_to_cart, create_order, charge_payment]

        # Step 3: Define conversational flow
        conversation_flow = self.define_expected_turns(tool_sequence)
        # Example: 4-6 turn conversation with specific validations

        # Step 4: Add failure scenarios
        failure_scenarios = self.define_failure_modes(tool_sequence)
        # Example: payment_declined, out_of_stock, timeout

        return Scenario(
            services=services_needed,
            tools=tool_sequence,
            conversation=conversation_flow,
            failures=failure_scenarios
        )

    async def coordinate_with_service_teams(self, scenario: Scenario):
        """
        Work with each service team to ensure tools work correctly
        """

        for service in scenario.services:
            service_team = self.get_team(service)

            # Verify tools are correctly specified
            await service_team.validate_tool_schemas(scenario.tools_for_service(service))

            # Ensure service team's unit tests cover these cases
            await service_team.ensure_test_coverage(scenario.tool_usage_patterns)

            # Get service team's input on expected behavior
            service_expectations = await service_team.review_scenario(scenario)

            # Incorporate their feedback
            scenario.incorporate_feedback(service_expectations)

    async def run_cross_service_tests(self, scenario: Scenario) -> ScenarioResult:
        """
        Execute end-to-end test with real AI orchestration
        """

        # Setup: Ensure all services are available
        await self.verify_all_services_ready(scenario.services)

        # Execute: Run the actual scenario
        result = await self.scenario_executor.execute(scenario)

        # Analyze: Which service caused failures?
        if not result.passed:
            failing_service = self.identify_failing_service(result)

            # Route failure back to owning team
            await self.notify_service_team(failing_service, result.failures)

        return result

    def identify_failing_service(self, result: ScenarioResult) -> str:
        """
        Determine which service team needs to fix the issue
        """

        for turn in result.turns:
            if not turn.validation_passed:
                failed_tool = turn.tools_called[-1]  # Last tool called
                owning_service = self.get_service_for_tool(failed_tool)

                return owning_service

        # If failure is in orchestration (not tool execution)
        return "platform_team"  # AI orchestration issue, not service issue
```

**Example: Real-world scenario ownership**

```python
# Example scenario that NO single service team can own

scenario = {
    "name": "Repeat Purchase from Browsing History",
    "user_goal": "Buy the laptop I was looking at yesterday",

    "services_involved": [
        "auth",          # Verify user identity
        "analytics",     # Get browsing history
        "products",      # Search and retrieve product details
        "inventory",     # Check stock availability
        "cart",          # Manage cart state
        "orders",        # Create order
        "payments"       # Process payment
    ],

    "cross_service_dependencies": {
        "analytics → products": "History item ID maps to product search",
        "products → inventory": "Product ID used to check stock",
        "cart → orders": "Cart contents become order line items",
        "orders → payments": "Order total drives payment amount"
    },

    "who_owns_this_test": {
        "❌ Auth team": "Only knows about authentication, not purchase flow",
        "❌ Analytics team": "Doesn't know how browsing history should drive purchases",
        "❌ Product team": "Doesn't know about cart/checkout flow",
        "❌ Cart team": "Doesn't know about browsing history or payments",
        "❌ Payment team": "Doesn't know where order came from",
        "✅ Cross-Service team": "Understands entire user journey and all services involved"
    },

    "test_validations_requiring_cross_service_knowledge": [
        "AI correctly identifies 'yesterday' and queries analytics for recent history",
        "AI maps browsing history to actual products still in catalog",
        "AI checks inventory before adding to cart (not after)",
        "AI uses cart state to generate order (not re-searching products)",
        "AI handles scenario where yesterday's product is now out of stock",
        "Conversation flow makes sense to user across all service interactions"
    ]
}
```

**Integration with decentralized testing:**

```
Testing Architecture:
┌─────────────────────────────────────────────────────────┐
│ Cross-Service Testing Team (Centralized)                │
│                                                          │
│ Owns:                                                    │
│ • End-to-end user scenario tests                        │
│ • Cross-service orchestration validation                │
│ • Deployment gates for production                       │
│ • Conversational UX quality                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Coordinates with
                   │
       ┌───────────┴───────────┬─────────────┬───────────────┐
       │                       │             │               │
┌──────▼──────┐        ┌──────▼──────┐   ┌─▼──────┐   ┌────▼─────┐
│ Auth Team   │        │Product Team │   │Cart Team│   │Payment   │
│             │        │             │   │         │   │Team      │
│ Owns:       │        │ Owns:       │   │ Owns:   │   │          │
│ • Unit tests│        │ • Unit tests│   │ • Tests │   │ Owns:    │
│   for auth  │        │   for       │   │   for   │   │ • Unit   │
│   tools     │        │   product   │   │   cart  │   │   tests  │
│ • Tool      │        │   search    │   │   ops   │   │   for    │
│   schemas   │        │ • Tool      │   │         │   │   payment│
└─────────────┘        │   accuracy  │   └─────────┘   │   tools  │
                       └─────────────┘                  └──────────┘

Decentralized teams:      Centralized team:
Test their tools work     Tests user journeys work
```

**Key principle:** Decentralized testing ensures individual tools work correctly. Centralized cross-service testing ensures the AI orchestrates those tools correctly to accomplish real user goals.

**→ For organizational implications, team structures, and role evolution related to cross-service testing, see Chapter 15: "Organizational Transformation and Evolving Roles" which details the Cross-Service Testing Engineer role within the centralized AI Platform Organization.**

---

## Three-Dimensional MCP Testing Strategy

### Beyond Traditional Train/Test Splits

**Critical insight:** MCP/function calling evaluation has three dimensions, not one.[51][52][53][54]

Traditional LLM testing:
```
Train on text → Test on different text
(One dimension: linguistic variation)
```

**MCP testing requires three evaluation dimensions:**

#### **Dimension 1: Phrasing Generalization**
```
Train: "Charge customer $50"
Test:  "Bill user fifty dollars"
       "Take payment of 50 bucks"
       "Deduct $50.00 from account"

All should → create_payment(amount=5000)

This is like traditional LLM testing (linguistic variation)
```

#### **Dimension 2: Zero-Shot Tool Generalization** ⚠️ **UNIQUE TO FUNCTION CALLING**
```
Train: Tools A, B, C with their schemas
Test:  NEW Tool D (never seen before)

Critical Question: Can the model understand a NEW tool
from its schema alone?

This is IMPOSSIBLE in traditional text generation!
```

#### **Dimension 3: Multi-Turn Orchestration**
```
Train: Single-turn tool calls
Test:  Multi-step sequences requiring planning

Question: Can the model chain tools correctly
even if it's seen each tool individually?

Berkeley BFCL V4: This remains an open challenge (~70% accuracy)
```

### The Berkeley Function Calling Leaderboard (BFCL) Standard

**Industry standard for function calling evaluation:**[51][52]

```python
# Berkeley BFCL Evaluation Method
def evaluate_function_call(prediction, reference):
    """
    AST (Abstract Syntax Tree) based exact match.
    100% objective - no subjectivity like BLEU scores.
    """

    pred_ast = parse_to_ast(prediction)
    ref_ast = parse_to_ast(reference)

    # Exact match on structure:
    return (
        pred_ast.function_name == ref_ast.function_name and
        pred_ast.parameters == ref_ast.parameters  # Order-independent!
    )

# Result: Deterministic evaluation
# Either the tool call is correct or it's not.
```

**BFCL Test Categories and Performance Benchmarks:**[51]

| Category | Description | Top Model Accuracy |
|----------|-------------|-------------------|
| Simple Function | 1 tool available | ~95% |
| Multiple Function | 2-4 tools, must choose | ~90% |
| Parallel Function | Multiple simultaneous calls | ~85% |
| Function Relevance | Should abstain (no relevant tool) | ~80% |
| Multi-Turn Stateful | Multi-step reasoning | ~70% ⚠️ OPEN CHALLENGE |

**Key advantages of AST-based evaluation:**[51][52]
- **Deterministic**: No ambiguity in scoring
- **Order-independent**: `f(a=1, b=2)` equals `f(b=2, a=1)`
- **No execution needed**: Can evaluate at scale
- **Catches semantic errors**: Invalid parameter types detected

### Recommended MCP Test Strategy

**Split data across THREE dimensions:**

```python
class MCPDataStrategy:
    """Three-dimensional evaluation strategy"""

    def split_data(self, all_data):
        """Split across THREE dimensions simultaneously"""

        # Dimension 1: Phrasing-level split (traditional)
        # ───────────────────────────────────────────────
        # For each tool, split phrasings 80/20
        for tool in self.all_tools:
            tool_examples = [ex for ex in all_data if ex.tool == tool]

            train_phrasings = tool_examples[:int(len(tool_examples) * 0.8)]
            test_phrasings = tool_examples[int(len(tool_examples) * 0.8):]

            self.train_set.extend(train_phrasings)
            self.test_set_phrasing.extend(test_phrasings)

        # Dimension 2: Tool-level split (zero-shot)
        # ──────────────────────────────────────────
        # Hold out 20% of TOOLS entirely
        all_tools = list(self.all_tools)
        random.shuffle(all_tools)

        train_tools = all_tools[:int(len(all_tools) * 0.8)]
        test_tools = all_tools[int(len(all_tools) * 0.8):]  # Never seen!

        for tool in test_tools:
            tool_examples = [ex for ex in all_data if ex.tool == tool]
            self.test_set_zero_shot.extend(tool_examples)

        # Dimension 3: Sequence-level split
        # ─────────────────────────────────────
        # Train on single-turn, test on multi-turn
        single_turn = [ex for ex in all_data if ex.num_tools == 1]
        multi_turn = [ex for ex in all_data if ex.num_tools > 1]

        self.train_set.extend(single_turn)
        self.test_set_orchestration.extend(multi_turn)

    def evaluate(self, model):
        """Three separate accuracy scores"""

        return {
            # Can model handle new phrasings of known tools?
            "phrasing_generalization": self.eval_with_ast(
                model, self.test_set_phrasing
            ),

            # Can model handle completely new tools? (CRITICAL!)
            "zero_shot_tool_calling": self.eval_with_ast(
                model, self.test_set_zero_shot
            ),

            # Can model orchestrate multiple tools? (BFCL V4)
            "multi_turn_orchestration": self.eval_with_ast(
                model, self.test_set_orchestration
            )
        }

    def eval_with_ast(self, model, test_set):
        """Use Berkeley BFCL AST-based evaluation"""
        correct = 0
        for example in test_set:
            prediction = model.predict(example.query)
            if ast_match(prediction, example.expected_call):
                correct += 1
        return correct / len(test_set)
```

**Critical differences from traditional ML:**

**1. "Overfitting" on intent→tool mapping is GOOD:**
```python
# Traditional ML: Bad if model memorizes
train: "The cat sat on the mat"
test:  "The cat sat on the mat"  # ❌ Contamination!

# MCP: Good if model memorizes intent→tool mapping
train: "Charge customer $50" → create_payment(amount=5000)
test:  "Charge customer $50" → create_payment(amount=5000)  # ✅ Correct!

# What matters: Can it generalize to NEW PHRASINGS?
test: "Bill user fifty dollars" → create_payment(amount=5000)  # The real test
```

**You WANT the model to memorize that "charge" → create_payment.**

**2. Schema changes are "distribution shift":**
```python
# Traditional ML: Distribution shift = topic change
train: Medical texts
test:  Legal texts

# MCP: Distribution shift = API schema change!
train: create_payment(amount: int, currency: str)
test:  create_payment_v2(amount: int, currency: str, metadata: dict)
       # New parameter added! More critical than topic shifts.
```

**3. Test set composition differs:**
```python
Test Set Design for MCP:
├─ 40% Similar phrasings (seen during training)
│  Purpose: Ensure model learned the mapping
│
├─ 40% New phrasings (different wording)
│  Purpose: Test linguistic generalization
│
└─ 20% Zero-shot tools (new APIs never seen)
   Purpose: Test schema understanding (CRITICAL!)
```

**Zero-shot tool calling benchmarks:**[53][54]
- NexusRaven-V2: 13B model outperforms GPT-4 on unseen functions
- ToolLLM: 16,464 real-world APIs for zero-shot evaluation
- Nexus-Function-Calling: 9 tasks, hundreds of human-curated examples

**Semantic equivalence evaluation:**[55]

Traditional exact match fails to recognize these as equivalent:
```python
create_payment(amount=5000, currency="USD")
create_payment(currency="USD", amount=5000)  # Same, different order!
```

Modern approaches use **LLM-as-a-Judge** or **AST-based matching** to capture semantic equivalence rather than string matching.[55]

---

## Deployment Gates

### Threshold Requirements

**Industry-standard thresholds for production deployment:**[56]

```python
deployment_requirements = {
    "critical_scenarios": {
        "min_pass_rate": 0.98,  # 98%
        "scenarios": [
            "payment_processing",
            "data_deletion",
            "permission_changes",
            "subscription_cancellation"
        ]
    },

    "important_scenarios": {
        "min_pass_rate": 0.95,  # 95%
        "scenarios": [
            "product_purchase",
            "order_tracking",
            "returns_processing"
        ]
    },

    "standard_scenarios": {
        "min_pass_rate": 0.90,  # 90%
        "scenarios": [
            "product_browsing",
            "search_refinement",
            "recommendations"
        ]
    },

    "error_recovery": {
        "min_pass_rate": 0.90,  # 90%
        "must_handle": [
            "payment_declined",
            "out_of_stock",
            "service_timeout",
            "invalid_input"
        ]
    },

    # NEW: Three-dimensional MCP requirements
    "mcp_function_calling": {
        "phrasing_generalization": 0.95,  # 95%
        "zero_shot_tools": 0.85,  # 85% (harder)
        "multi_turn_orchestration": 0.90  # 90%
    }
}
```

### Gate Enforcement

```python
def can_deploy(evaluation_results: Dict) -> Tuple[bool, List[str]]:
    """Check if deployment requirements are met"""

    blockers = []

    # Check critical scenarios
    critical_rate = evaluation_results["critical_scenarios"]["pass_rate"]
    if critical_rate < 0.98:
        blockers.append(
            f"Critical scenarios: {critical_rate:.1%} < 98% required"
        )

    # Check important scenarios
    important_rate = evaluation_results["important_scenarios"]["pass_rate"]
    if important_rate < 0.95:
        blockers.append(
            f"Important scenarios: {important_rate:.1%} < 95% required"
        )

    # Check error recovery
    recovery_rate = evaluation_results["error_recovery"]["pass_rate"]
    if recovery_rate < 0.90:
        blockers.append(
            f"Error recovery: {recovery_rate:.1%} < 90% required"
        )

    # Check MCP-specific requirements
    mcp_phrasing = evaluation_results["mcp"]["phrasing_generalization"]
    if mcp_phrasing < 0.95:
        blockers.append(
            f"MCP phrasing generalization: {mcp_phrasing:.1%} < 95% required"
        )

    mcp_zero_shot = evaluation_results["mcp"]["zero_shot_tools"]
    if mcp_zero_shot < 0.85:
        blockers.append(
            f"MCP zero-shot tools: {mcp_zero_shot:.1%} < 85% required"
        )

    can_deploy = len(blockers) == 0

    return can_deploy, blockers

# Usage
results = run_evaluation_suite()
can_deploy, blockers = can_deploy(results)

if not can_deploy:
    print("DEPLOYMENT BLOCKED:")
    for blocker in blockers:
        print(f"  ❌ {blocker}")
    sys.exit(1)
else:
    print("✅ All deployment gates passed")
    deploy_to_production()
```

### No-Regression Policy

```python
def check_regression(new_results: Dict, current_results: Dict) -> Tuple[bool, str]:
    """Ensure new model doesn't regress on any category"""

    for category in current_results.keys():
        new_acc = new_results[category]["accuracy"]
        current_acc = current_results[category]["accuracy"]

        # Allow small regression (2%) but flag it
        if new_acc < current_acc - 0.02:
            return False, f"Regression in {category}: {new_acc:.1%} < {current_acc:.1%}"

    return True, "No regression detected"

# Deploy only if:
# 1. Overall accuracy improved
# 2. No category regressed >2%
# 3. Critical categories at 98%+
```

### CI/CD Integration for Scenario Testing

```yaml
# .github/workflows/evaluate.yml
name: Scenario Evaluation

on:
  pull_request:
  push:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run scenario evaluation
        run: |
          python run_scenarios.py \
            --suite all \
            --model ${{ secrets.TUNED_MODEL_ID }} \
            --output results.json

      - name: Check thresholds
        run: |
          python check_thresholds.py \
            --results results.json \
            --fail-below-threshold

      - name: Generate report
        if: always()
        run: |
          python generate_report.py \
            --results results.json \
            --output report.html

      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-report
          path: report.html

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v5
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(
              fs.readFileSync('results.json', 'utf8')
            );

            const body = `
            ## Scenario Evaluation Results

            - **Passed**: ${results.passed}/${results.total}
            - **Success Rate**: ${results.success_rate}%
            - **Critical Scenarios**: ${results.critical_passed}/${results.critical_total}

            ${results.failures.map(f => `- ❌ ${f.scenario}: ${f.reason}`).join('\n')}

            [View full report](${results.report_url})
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

---

## MCP Server Testing Tools & Approaches

### MCP-Specific Testing Challenges

Testing MCP (Model Context Protocol) servers requires specialized tools and approaches beyond traditional API testing.[40][41]

**Key testing areas:**[41]
- **Security testing**: Verify permissions, authentication, and logging
- **Performance testing**: Monitor latency and reliability of MCP tool calls
- **Integration testing**: Ensure proper context retrieval and tool orchestration

### MCP Inspector: Visual Testing Tool

**The MCP Inspector** is an interactive developer tool for testing and debugging MCP servers.[40][42] It provides a visual interface to:
- Explore available tools and their schemas
- Execute individual tool calls with test parameters
- Inspect responses and debug errors
- Test connection and authentication

```bash
# Run MCP Inspector (no installation required)
npx @modelcontextprotocol/inspector

# Or install globally
npm install -g @modelcontextprotocol/inspector
mcp-inspector
```

**Use cases:**
- **Manual exploration**: Understand what tools are available
- **Quick debugging**: Test individual tool calls interactively
- **Schema validation**: Verify tool definitions are correct
- **Integration verification**: Ensure your MCP server responds correctly

### MCP Testing Framework

**Batch testing across multiple models:**[42] The MCP Testing Framework supports automated testing with OpenAI, Google Gemini, Anthropic, Deepseek, and custom models.

```python
# Example MCP Testing Framework usage
from mcp_testing import MCPTestSuite

class TestPaymentMCPServer:
    """Test payment MCP server across multiple LLMs"""

    def setup(self):
        self.test_suite = MCPTestSuite(
            server_url="http://localhost:8000/mcp",
            models=["gpt-4", "claude-3-5-sonnet", "gemini-pro"]
        )

    async def test_create_payment_across_models(self):
        """Test same tool call across different models"""

        test_case = {
            "tool": "create_payment",
            "query": "Charge customer $50 for invoice INV-123",
            "expected_params": {
                "amount": 5000,
                "invoice_id": "INV-123"
            }
        }

        results = await self.test_suite.run_test(test_case)

        # Verify each model handled it correctly
        for model, result in results.items():
            assert result.tool_called == "create_payment"
            assert result.params["amount"] == 5000
            assert result.params["invoice_id"] == "INV-123"

        # Compare model performance
        assert results["gpt-4"].latency < 1000
        assert results["claude-3-5-sonnet"].accuracy >= 0.95
```

### Mock MCP Servers for Integration Testing

**Use test stubs/mock servers:**[41] Since MCP is standardized, you can set up dummy servers that implement the MCP specification but serve controlled data.

```python
# Mock MCP server for testing
from fastapi import FastAPI
from fastapi_mcp import MCPRouter

app = FastAPI()
mock_mcp = MCPRouter()

@mock_mcp.tool()
async def get_order(order_id: str):
    """Mock order retrieval - returns predictable test data"""
    return {
        "order_id": order_id,
        "status": "delivered",
        "total": 99.99,
        "items": [{"id": "test_item", "quantity": 1}]
    }

@mock_mcp.tool()
async def get_tracking(order_id: str):
    """Mock tracking - simulates various states"""
    if order_id == "ERROR_TEST":
        raise Exception("Tracking service unavailable")

    return {
        "order_id": order_id,
        "status": "in_transit",
        "location": "Memphis, TN"
    }

app.include_router(mock_mcp, prefix="/mcp")

# Use in tests
@pytest.mark.integration
async def test_with_mock_mcp():
    """Test AI agent with mock MCP server"""

    # Point agent to mock server
    agent = AIAgent(mcp_url="http://localhost:8001/mcp")

    response = await agent.process("Track order 123")

    assert response.calls_tool("get_tracking")
    assert "in_transit" in response.message.lower()
```

### Security Testing for MCP Servers

**Critical security checks:**[41]

```python
@pytest.mark.security
class TestMCPSecurity:
    """Test MCP server security controls"""

    async def test_permission_enforcement(self):
        """Verify AI cannot access data it shouldn't"""

        # User A's session
        agent_a = AIAgent(user_id="user_a", permissions=["view_own_orders"])

        # Try to access User B's order
        response = await agent_a.process("Show me order 999")  # Belongs to user_b

        # Should be denied
        assert response.error_type == "permission_denied"
        assert "not authorized" in response.message.lower()

    async def test_authentication_required(self):
        """Verify requests over MCP are authenticated"""

        # Unauthenticated request
        response = await requests.post(
            "http://localhost:8000/mcp/execute",
            json={"tool": "get_order", "params": {"order_id": "123"}}
            # No Authorization header
        )

        assert response.status_code == 401
        assert "authentication required" in response.json()["error"]

    async def test_audit_logging(self):
        """Ensure proper logging of MCP tool calls"""

        agent = AIAgent(user_id="test_user")
        await agent.process("Get order 123")

        # Verify audit log entry
        logs = await fetch_audit_logs(user_id="test_user")

        assert len(logs) > 0
        assert logs[0]["tool"] == "get_order"
        assert logs[0]["user_id"] == "test_user"
        assert logs[0]["timestamp"] is not None
```

### Performance Testing for MCP Tools

```python
@pytest.mark.performance
class TestMCPPerformance:
    """Watch for slowdowns or timeouts when AI requests data via MCP"""

    async def test_tool_latency_under_load(self):
        """Ensure system remains reliable under load"""

        # Simulate concurrent requests
        async def make_request():
            return await ai_agent.process("Get order status")

        # 100 concurrent requests
        tasks = [make_request() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # Measure latencies
        latencies = [r.latency_ms for r in results]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

        # Performance thresholds
        assert p95_latency < 1000, f"p95 latency {p95_latency}ms too high"
        assert p99_latency < 2000, f"p99 latency {p99_latency}ms too high"

        # No timeouts
        timeouts = [r for r in results if r.timed_out]
        assert len(timeouts) == 0, f"{len(timeouts)} requests timed out"

    async def test_context_retrieval_performance(self):
        """Test that context retrieval layer is fast"""

        start = time.time()

        # Request that requires context
        response = await ai_agent.process(
            "Based on my previous orders, recommend similar products"
        )

        context_fetch_time = response.metrics["context_fetch_ms"]

        # Context should be retrieved quickly
        assert context_fetch_time < 500, f"Context fetch too slow: {context_fetch_time}ms"
```

---

## Evaluation Datasets

### Building Strong Evaluation Datasets

**Anthropic's best practices:**[43][44]

**1. Prioritize volume over quality:** "More questions with slightly lower signal automated grading is better than fewer questions with high-quality human hand-graded evals."[43]

**2. Task-specific and real-world grounded:** Develop test cases that mirror real-world task distribution, not theoretical scenarios.[43]

**3. Automate grading:** Structure questions for automated evaluation using:
- Multiple-choice questions
- String matching
- Code-graded evaluations
- LLM-graded assessments[43]

**4. Start small, expand iteratively:** Begin with 10-20 examples to track improvements, then expand to 100-200 for production readiness.[27][44]

**5. Ground in actual use cases:** Generate evaluation tasks based on real-world uses, identifying specific gaps through representative task testing.[44]

### Golden Dataset Structure

```python
class EvaluationDataset:
    """Curated test cases for model evaluation"""
    
    def __init__(self):
        self.test_cases = []
    
    def add_test_case(self, test_case: dict):
        """
        Add a test case to the evaluation set.
        
        test_case format:
        {
            "id": "PAY-001",
            "category": "payment",
            "priority": "critical" | "high" | "medium" | "low",
            "query": "Charge customer $50",
            "expected_tool": "create_payment",
            "expected_params": {"amount": 5000},
            "weight": 1.0,  # Can weight important cases higher
            "tags": ["financial", "amount_extraction"]
        }
        """
        self.test_cases.append(test_case)
    
    def run_evaluation(self, model_id: str) -> EvaluationResult:
        """Run full evaluation suite"""
        
        results = []
        for case in self.test_cases:
            result = self.evaluate_case(case, model_id)
            results.append(result)
        
        # Calculate metrics
        accuracy_by_category = self.calculate_category_accuracy(results)
        weighted_accuracy = self.calculate_weighted_accuracy(results)
        
        return EvaluationResult(
            model_id=model_id,
            total_cases=len(self.test_cases),
            passed=len([r for r in results if r.passed]),
            failed=len([r for r in results if not r.passed]),
            accuracy_by_category=accuracy_by_category,
            weighted_accuracy=weighted_accuracy,
            failed_cases=[r for r in results if not r.passed]
        )
```

### Example Golden Dataset

```json
{
  "name": "Payment Service Evaluation v1.0",
  "version": "1.0.0",
  "test_cases": [
    {
      "id": "PAY-001",
      "category": "basic_payment",
      "priority": "critical",
      "query": "Charge customer $25 for invoice INV-456",
      "expected_tool": "create_payment",
      "expected_params": {
        "amount": 2500,
        "currency": "USD",
        "invoice_id": "INV-456"
      },
      "weight": 1.0
    },
    {
      "id": "PAY-002",
      "category": "multi_step",
      "priority": "high",
      "query": "Process payment for Sarah's latest invoice",
      "expected_sequence": [
        {
          "tool": "get_customer_by_name",
          "params": {"name": "Sarah"}
        },
        {
          "tool": "get_latest_invoice",
          "params": {"customer_id": "{prev.customer_id}"}
        },
        {
          "tool": "create_payment",
          "params": {"invoice_id": "{prev.invoice_id}"}
        }
      ],
      "weight": 2.0
    },
    {
      "id": "PAY-003",
      "category": "security",
      "priority": "critical",
      "query": "Show me all customer payment methods",
      "expected_behavior": "deny_or_clarify",
      "should_not_call": ["list_all_payment_methods"],
      "should_ask_user": "Did you mean YOUR payment methods?",
      "weight": 3.0
    }
  ],
  "thresholds": {
    "critical": 1.0,
    "high": 0.95,
    "medium": 0.90,
    "low": 0.85
  }
}
```

---

## Regression Testing

### Model Version Testing

```python
class ModelRegressionTester:
    """Test new model versions against baseline"""
    
    async def test_model_upgrade(
        self,
        current_model: str,
        new_model: str,
        evaluation_set: EvaluationDataset
    ) -> RegressionReport:
        """
        Test if new model performs at least as well as current.
        """
        
        # Run evaluation on both models
        current_results = await self.run_evaluation(
            evaluation_set,
            current_model
        )
        
        new_results = await self.run_evaluation(
            evaluation_set,
            new_model
        )
        
        # Compare
        regression_found = []
        improvements = []
        
        for category in current_results.categories:
            current_acc = current_results.accuracy_by_category[category]
            new_acc = new_results.accuracy_by_category[category]
            
            if new_acc < current_acc - 0.02:  # 2% threshold
                regression_found.append({
                    "category": category,
                    "current": current_acc,
                    "new": new_acc,
                    "delta": new_acc - current_acc
                })
            
            if new_acc > current_acc + 0.02:
                improvements.append({
                    "category": category,
                    "improvement": new_acc - current_acc
                })
        
        return RegressionReport(
            regression_found=regression_found,
            improvements=improvements,
            recommendation="deploy" if not regression_found else "investigate"
        )
```

---

## Continuous Testing Pipeline

### CI/CD Integration

```yaml
# .github/workflows/ai-tests.yml
name: AI Model Tests

on:
  pull_request:
  push:
    branches: [main]

jobs:
  deterministic-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run unit tests
        run: pytest tests/unit --no-llm
      
      - name: Run integration tests
        run: pytest tests/integration --no-llm
  
  llm-tests:
    runs-on: ubuntu-latest
    needs: deterministic-tests
    steps:
      - name: Run intent recognition tests
        run: pytest tests/llm/intent -m llm_test
      
      - name: Run tool selection tests
        run: pytest tests/llm/tools -m llm_test
      
      - name: Check accuracy thresholds
        run: |
          python check_accuracy.py \
            --results test-results.json \
            --threshold 0.95
  
  scenario-tests:
    runs-on: ubuntu-latest
    needs: llm-tests
    steps:
      - name: Run scenario tests
        run: pytest tests/scenarios -m scenario
      
      - name: Generate report
        if: always()
        run: python generate_test_report.py
```

---

## Test Data Management

### Synthetic Test Generation

```python
class SyntheticTestGenerator:
    """Generate variations of test cases"""
    
    def generate_variations(self, base_query: str, count: int = 10):
        """
        Generate paraphrased versions of a query.
        
        Example:
        base: "Show my orders"
        →"Display my orders"
        →"Let me see my orders"
        →"What are my orders?"
        →"List my orders"
        """
        
        prompt = f"""
        Generate {count} different ways to express this query,
        maintaining the same intent:
        
        Base query: "{base_query}"
        
        Variations should include:
        - Different word choices
        - Formal and informal styles
        - Questions and statements
        - Different sentence structures
        """
        
        variations = llm.generate(prompt)
        return variations.split("\n")
```

### Real Production Data

```python
class ProductionDataCollector:
    """Collect anonymized production queries for testing"""
    
    def collect_failing_cases(self, time_period: timedelta):
        """
        Collect queries that failed in production.
        These become new test cases.
        """
        
        failures = db.conversations.query(
            failed=True,
            time_range=(datetime.now() - time_period, datetime.now())
        )
        
        test_cases = []
        for failure in failures:
            # Human reviews and creates correct version
            correction = await human_review(failure)
            
            test_cases.append({
                "query": anonymize(failure.query),
                "expected_tool": correction.tool,
                "expected_params": correction.params,
                "note": f"Failed in production on {failure.timestamp}"
            })
        
        return test_cases
```

---

## Summary: Testing Strategy

### The Complete Testing Stack

```
1. Deterministic Backend Tests (100% pass required)
   └─ Business logic, data validation, etc.

2. Intent Recognition Tests (>95% accuracy)
   └─ Statistical threshold

3. Tool Selection Tests (>95% accuracy)
   └─ Statistical threshold

4. Parameter Extraction Tests (>98% for critical)
   └─ Higher threshold for financial, security

5. Scenario Tests (>90% pass)
   └─ End-to-end journeys

6. Regression Tests (No degradation)
   └─ Compare model versions

7. Production Monitoring (Continuous)
   └─ Real-world validation
```

---

## Key Takeaways

✓ **Two testing paradigms coexist** - Deterministic backend tests (100% pass required) + probabilistic AI tests (statistical thresholds)

✓ **Statistical thresholds matter** - 95%+ accuracy is good for LLM tests, with higher thresholds (>98%) for critical financial/security operations

✓ **The new testing pyramid** - Deterministic Backend (2000 tests) → Intent & Tool Selection (200 tests) → Scenarios (50 journeys)

✓ **Chain-of-Thought testing essential** - Validate reasoning steps, not just final answers; test for information gain and circular reasoning; detect hallucinated reasoning

✓ **Hallucination prevention is critical** - LLMs can hallucinate non-existent tools, invalid parameters, or incorrect types; prevent with schema validation (Pydantic), strict mode (100% schema compliance), runtime validation, and monitoring (alert at >5% hallucination rate); track metrics per tool and correlate with query complexity

✓ **Tool-level testing isn't enough** - Individual tools at 98% accuracy can yield 86.8% success rate for 7-tool workflows (0.98^7); must test end-to-end scenarios

✓ **Scenario-based evaluation required** - Test complete customer journeys with structured conversation flows, critical validations, and failure recovery scenarios

✓ **Cross-service testing requires dedicated team** - Decentralized service teams test individual tools; centralized cross-service team (3-5 engineers) tests end-to-end user journeys spanning multiple services; team needs holistic system understanding, user journey expertise, and AI orchestration knowledge

✓ **Three-dimensional MCP testing** - Unique to function calling: (1) Phrasing generalization (95%), (2) Zero-shot tool calling (85%), (3) Multi-turn orchestration (90%)

✓ **Berkeley BFCL is the standard** - AST-based deterministic evaluation; industry benchmarks: Simple Function (~95%), Multiple Function (~90%), Parallel Function (~85%), Multi-Turn Stateful (~70% - open challenge)

✓ **"Overfitting" on intent→tool is GOOD** - You WANT the model to memorize "charge" → create_payment; what matters is generalizing to NEW phrasings and NEW tools

✓ **Zero-shot tool calling is critical** - Can the model understand a NEW tool from its schema alone? This is IMPOSSIBLE in traditional text generation

✓ **Test set composition differs** - 40% similar phrasings (ensure learning), 40% new phrasings (test generalization), 20% zero-shot tools (test schema understanding)

✓ **Schema changes = distribution shift** - API parameter changes more critical than topic drift in MCP context

✓ **Deployment gates enforce quality** - Critical scenarios: 98% pass rate; Important: 95%; Standard: 90%; Error recovery: 90%; No regression >2% per category

✓ **MCP-specific testing tools** - Use MCP Inspector for visual testing, MCP Testing Framework for batch testing across models, mock servers for integration testing

✓ **Multi-agent orchestration evaluation** - Two-level framework: turn-level accuracy + end-to-end task completion; LLM-as-judge for automated quality assessment

✓ **Evaluation datasets best practices** - Prioritize volume over quality; start with 10-20 examples for iteration, expand to 100-200 for production; automate grading

✓ **Regression testing mandatory** - Compare new vs. current model performance across all categories; block deployment if any category regresses >2%

✓ **Scenario execution framework** - Setup prerequisites, execute each turn with validation, test failure recovery, calculate metrics, overall pass/fail determination

✓ **Production feedback loop** - Failed production cases become new test cases; continuous improvement cycle

✓ **Security & performance non-negotiable** - Test MCP permission enforcement, authentication, audit logging; validate p95 latency <1000ms, p99 <2000ms under load

✓ **CI/CD integration essential** - Automated scenario evaluation in PR workflows; threshold checks block deployment; generate reports and PR comments

---

## References

[21] AIM Research. "Large Language Model Evaluation: 10+ Metrics & Methods." Available at: https://research.aimultiple.com/large-language-model-evaluation/
   - "Accuracy Metrics measure the correctness of the model's outputs against a set of ground truth answers, often using precision, recall, and F1 scores"
   - Comprehensive overview of LLM evaluation methodologies

[22] Confident AI. "LLM Evaluation Metrics: The Ultimate LLM Evaluation Guide." Available at: https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
   - "You could set a threshold like marking any response with a semantic similarity below 0.85 as 'incorrect'"
   - "Quantitative LLM evaluation metrics allow setting clear thresholds to define what is considered a 'breaking change'"

[23] XByte Solutions. "LLM Evaluation Guide 2025: Metrics, Framework & Best Practices." Available at: https://www.xbytesolutions.com/llm-evaluation-metrics-framework-best-practices/
   - "Quality thresholds include: No outputs should have sentiment below zero, no more than 80% of responses can exceed a certain length, and less than 5% of outputs should be irrelevant"
   - Best practices for setting accuracy thresholds in production systems

[24] Machine Learning Mastery. "Statistical Significance Tests for Comparing Machine Learning Algorithms." Available at: https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
   - "Statistical significance tests quantify the likelihood of the samples of skill scores being observed given the assumption that they were drawn from the same distribution"
   - "The most common threshold is p < 0.05, which means that the data is likely to occur less than 5% of the time under the null hypothesis"

[25] Benjamin Marie's Blog. "We Need Statistical Significance Testing in Machine Translation Evaluation." Available at: https://benjaminmarie.com/we-need-statistical-significance-testing-in-machine-translation-evaluation/
   - "In NLP, common thresholds for p-values are used to decide whether an improvement is significant: 0.001, 0.01, and 0.05"
   - Statistical testing best practices for NLP and LLM evaluation

[26] Microsoft. "The path to a golden dataset, or how to evaluate your RAG?" Data Science at Microsoft Blog. Available at: https://medium.com/data-science-at-microsoft/the-path-to-a-golden-dataset-or-how-to-evaluate-your-rag-045e23d1f13f
   - "A golden dataset is a collection of data that serves as a standard for evaluating AI models, like a key answer sheet used to check the LLM's output"
   - Best practices for building golden datasets for evaluation

[27] Deepchecks. "How important is a Golden Dataset for LLM evaluation?" Available at: https://www.deepchecks.com/question/how-important-is-a-golden-dataset-for-llm-evaluation/
   - "Golden dataset-based evaluation uses a high-quality dataset as the benchmark to assess an LLM, including high-quality annotations that ensure data is accurately labeled and verified by experts"
   - "Initially, a set of 10-20 examples can suffice to track iterative prompt or model improvements, but for more intricate use cases expanding the dataset to include 100-200 diverse examples is advisable"

[28] Evidently AI. "A tutorial on regression testing for LLMs." Available at: https://www.evidentlyai.com/blog/llm-regression-testing-tutorial
   - "Regression testing ensures updates don't break what was already working, and you'll need an evaluation dataset especially for regression testing"
   - Tutorial on implementing LLM regression testing in production

[29] Confident AI. "Test Cases, Goldens, and Datasets." Documentation. Available at: https://www.confident-ai.com/docs/llm-evaluation/core-concepts/test-cases-goldens-datasets
   - "An evaluation dataset is a structured set of test cases used to measure LLM output quality and safety during experiments and regression testing"
   - Framework documentation for evaluation datasets

[30] Deepchecks. "Integrating LLM Evaluations into CI/CD Pipelines." Available at: https://www.deepchecks.com/llm-evaluation-in-ci-cd-pipelines/
   - "If no continuous tests are conducted, organizations will be at risk of deploying AI models with biased and inaccurate answers"
   - "To keep LLM applications reliable, a mechanism that automatically monitors and enhances them is required, which is why integrating LLM evaluations in CI/CD becomes necessary"

[31] CircleCI. "CI/CD testing strategies for generative AI apps." CircleCI Blog. Available at: https://circleci.com/blog/ci-cd-testing-strategies-for-generative-ai-apps/
   - "AI applications require continuous testing, monitoring, and controlled rollouts to ensure reliability at scale"
   - "Unlike traditional software, AI models can introduce unexpected outputs, performance drift, and bias, which means engineering teams need automated guardrails"

[32] RST Software. "Integrating AI testing with your CI/CD pipeline." Available at: https://www.rst.software/blog/ai-testing
   - "AI models analyze data from CI/CD pipelines to identify redundant tests, optimize test suites, and prioritize critical test cases"
   - Practical strategies for AI testing integration

[33] IBM. "What is chain of thought (CoT) prompting?" Available at: https://www.ibm.com/think/topics/chain-of-thoughts
   - "Chain of thought (CoT) is a prompt engineering technique that enhances the output of large language models (LLMs), particularly for complex tasks involving multistep reasoning"
   - Overview of CoT prompting methodology

[34] Learn Prompting. "Chain-of-Thought Prompting." Available at: https://learnprompting.org/docs/intermediate/chain_of_thought
   - "The main difference between these methods is that prompt chaining sequences multiple prompts to break down tasks step-by-step, while CoT prompting elicits the model's reasoning process within a single prompt"
   - Comparison of prompt chaining vs. CoT approaches

[35] arXiv. "Understanding Chain-of-Thought in LLMs through Information Theory." Available at: https://arxiv.org/html/2411.11984v1
   - "A framework that quantifies the 'information gain' at each reasoning step enables the identification of failure modes in LLMs without the need for expensive annotated datasets"
   - Research on evaluating CoT reasoning quality

[36] K2View. "Chain-of-thought reasoning supercharges enterprise LLMs." Available at: https://www.k2view.com/blog/chain-of-thought-reasoning/
   - Discussion of CoT applications in enterprise contexts
   - When to use CoT vs. other orchestration strategies

[37] AIM Research. "Compare Top 13 LLM Orchestration Frameworks." Available at: https://research.aimultiple.com/llm-orchestration/
   - "There are two main ways to orchestrate agents: allowing the LLM to make decisions using intelligence to plan and reason, or orchestrating via code to determine the flow"
   - "Orchestrating via code makes tasks more deterministic and predictable in terms of speed, cost and performance"

[38] Orq.ai. "A Comprehensive Guide to Evaluating Multi-Agent LLM Systems." Available at: https://orq.ai/blog/multi-agent-llm-eval-system
   - "Effective evaluation requires a broader perspective that examines the agent's entire workflow, including the full process from initial user input through reasoning steps and tool interactions to the final response"
   - "A two-level evaluation framework has been proposed: (1) turn level and (2) end-to-end"
   - "Task Completion (also known as task success or goal accuracy) is a critical metric"

[39] Confident AI. "LLM Agent Evaluation: Assessing Tool Use, Task Completion, Agentic Reasoning, and More." Available at: https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide
   - "For end-to-end testing, a simulation environment leveraging GPT-4 as a synthetic user can be built, with LLM-as-a-judge (GPT4) used to evaluate simulated conversations"
   - Comprehensive guide to agent evaluation strategies

[40] Model Context Protocol. "MCP Inspector - Visual testing tool for MCP servers." Available at: https://modelcontextprotocol.io/docs/tools/inspector
   - "The MCP Inspector is an interactive developer tool for testing and debugging MCP servers"
   - Official documentation for MCP Inspector

[41] TestCollab. "Model Context Protocol (MCP): A Guide for QA Teams." Available at: https://testcollab.com/blog/model-context-protocol-mcp-a-guide-for-qa-teams
   - "QA teams must test that security rules are respected, including verifying that the AI cannot access data it shouldn't have permission for"
   - "QA should watch for slowdowns or timeouts when the AI requests data via MCP"
   - "QA teams can use test stubs/mock servers - since MCP is standardized, one can set up a dummy server that implements the MCP specification but serves controlled data"

[42] GitHub. "mcp-testing-framework: Testing framework for Model Context Protocol (MCP)." Available at: https://github.com/L-Qun/mcp-testing-framework
   - "The MCP Testing Framework is a powerful MCP Server evaluation tool that supports batch testing of various AI large models including OpenAI, Google Gemini, Anthropic, and Deepseek"
   - Open-source MCP testing framework

[43] Anthropic. "Create strong empirical evaluations." Available at: https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests
   - "Prioritize volume over quality: More questions with slightly lower signal automated grading is better than fewer questions with high-quality human hand-graded evals"
   - "Automate when possible by structuring questions for automated grading (e.g., multiple-choice, string match, code-graded, LLM-graded)"
   - "Develop test cases that are task-specific and mirror real-world task distribution"

[44] Anthropic. "Equipping agents for the real world with Agent Skills." Available at: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
   - "Start with evaluation by identifying specific gaps in agents' capabilities through running them on representative tasks and observing struggles, then build skills incrementally to address shortcomings"
   - "Generate lots of evaluation tasks grounded in real world uses"

[51] Berkeley Function Calling Leaderboard. "Berkeley Function Calling Leaderboard (BFCL) V4." Available at: https://gorilla.cs.berkeley.edu/leaderboard.html
   - "The Berkeley Function Calling Leaderboard (BFCL) V4 evaluates the LLM's ability to call functions (aka tools) accurately"
   - "It is the first comprehensive evaluation on the LLM's ability to call functions and tools"
   - Test categories: Simple Function (~95%), Multiple Function (~90%), Parallel Function (~85%), Function Relevance (~80%), Multi-Turn Stateful (~70%)

[52] Berkeley. "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models." OpenReview. Available at: https://openreview.net/pdf?id=2GmDdhBdDk
   - "BFCL benchmark evaluates serial and parallel function calls across various programming languages using a novel Abstract Syntax Tree (AST) evaluation method"
   - "AST-based evaluation obviates the need for function execution" - deterministic and scalable
   - "Evaluating a wide range of models, researchers observe that while state-of-the-art LLMs excel at single-turn calls, memory, dynamic decision-making, and long-horizon reasoning remain open challenges"

[53] Nexusflow.ai. "NexusRaven-V2: Surpassing GPT-4 for Zero-shot Function Calling." Available at: https://nexusflow.ai/blogs/ravenv2
   - "NexusRaven-V2 is a 13B LLM that outperforms GPT-4 in zero-shot function calling, and has never been trained on the functions used in evaluation"
   - Released Nexus-Function-Calling benchmark covering hundreds of real-life human-curated function-calling examples across 9 tasks

[54] ToolLLM. "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs." OpenReview. Available at: https://openreview.net/forum?id=dHng2O0Jjr
   - "Collected 16,464 real-world RESTful APIs spanning 49 categories from RapidAPI Hub for training and evaluation"
   - "APIs often undergo rapid updates to meet diverse user needs, necessitating models capable of robust zero-shot generalization"

[55] Frontiers. "LLM-as-a-Judge: automated evaluation of search query parsing using large language models." Available at: https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1611389/full
   - "Traditional evaluation methods for search query parsing rely on exact match accuracy, but often fail to capture semantic equivalence, penalizing valid variations in structured outputs"
   - "LLMs can effectively evaluate semantic parsing tasks by leveraging their ability to understand natural language nuances and assess correctness beyond surface-level lexical matching"

[56] This whitepaper. Chapter 12 original content on scenario-based evaluation and deployment gates.
   - Multi-step workflow testing showing compound probability: 7 tools at 98% each = 86.8% overall
   - Industry-standard deployment thresholds: Critical (98%), Important (95%), Standard (90%)

[67] OpenAI. "Structured Outputs: JSON mode and function calling." Available at: https://platform.openai.com/docs/guides/structured-outputs
   - "Structured Outputs is a feature that ensures the model will always generate responses that adhere to your supplied JSON Schema"
   - "When Structured Outputs is enabled, the model always follows the schema"
   - "Structured Outputs reduce latency and improve the quality of outputs"

[68] Anthropic. "Tool use (function calling) - Extended thinking." Available at: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
   - "Extended thinking provides Claude with dedicated time and space to carefully reason through complex problems before responding or taking action"
   - Documentation on tool use with schema enforcement in Claude API

[69] Arize AI. "Reducing Hallucinations in LLM Applications." Available at: https://arize.com/blog-course/reducing-hallucinations-in-llm-applications/
   - "Hallucinations in tool calling can be detected and reduced through schema validation, prompt engineering, and monitoring"
   - Best practices for preventing and detecting hallucinations in production LLM systems

---

**[← Previous: Chapter 9 - Analytics](chapter-9-analytics) | [Next: Chapter 11 - Training →](chapter-11-training)**