import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant
from fle.eval.inspect_integration.transforms import middle_out, estimate_tokens

def test_estimate_tokens():
    msg = ChatMessageUser(content="1234")
    assert estimate_tokens(msg) == 1
    
    msg = ChatMessageUser(content="12345678")
    assert estimate_tokens(msg) == 2
    
    msg = ChatMessageUser(content="")
    assert estimate_tokens(msg) == 0

def test_middle_out_no_truncation():
    messages = [
        ChatMessageSystem(content="System"),
        ChatMessageUser(content="User 1"),
        ChatMessageAssistant(content="Assistant 1")
    ]
    # Total tokens roughly: 6/4 + 6/4 + 11/4 = 1 + 1 + 2 = 4
    result = middle_out(messages, max_tokens=100)
    assert len(result) == 3
    assert result == messages

def test_middle_out_truncation():
    # Create messages with predictable token counts
    # "1234" = 1 token
    
    system = ChatMessageSystem(content="1234") # 1 token
    user1 = ChatMessageUser(content="1234") # 1 token
    assist1 = ChatMessageAssistant(content="1234") # 1 token
    user2 = ChatMessageUser(content="1234") # 1 token
    assist2 = ChatMessageAssistant(content="1234") # 1 token
    user3 = ChatMessageUser(content="1234") # 1 token
    
    messages = [system, user1, assist1, user2, assist2, user3]
    # Total 6 tokens
    
    # Limit to 3 tokens
    # Should keep system (1) + user3 (1) + assist2 (1) = 3 tokens
    # Wait, logic is: keep system, then fill from back.
    # So: system (1), then user3 (1), then assist2 (1). Total 3.
    
    result = middle_out(messages, max_tokens=3)
    
    assert len(result) == 3
    assert result[0] == system
    assert result[1] == assist2
    assert result[2] == user3

def test_middle_out_truncation_exact_limit():
    system = ChatMessageSystem(content="1234") # 1 token
    user1 = ChatMessageUser(content="1234") # 1 token
    
    messages = [system, user1]
    result = middle_out(messages, max_tokens=2)
    assert len(result) == 2
    assert result == messages

def test_middle_out_truncation_system_only():
    system = ChatMessageSystem(content="1234") # 1 token
    user1 = ChatMessageUser(content="1234") # 1 token
    
    messages = [system, user1]
    result = middle_out(messages, max_tokens=1)
    # Should keep system only
    assert len(result) == 1
    assert result[0] == system

def test_middle_out_no_system():
    user1 = ChatMessageUser(content="1234") # 1 token
    user2 = ChatMessageUser(content="1234") # 1 token
    user3 = ChatMessageUser(content="1234") # 1 token
    
    messages = [user1, user2, user3]
    
    # Limit to 2 tokens
    # Should keep user3, user2
    result = middle_out(messages, max_tokens=2)
    
    assert len(result) == 2
    assert result[0] == user2
    assert result[1] == user3
