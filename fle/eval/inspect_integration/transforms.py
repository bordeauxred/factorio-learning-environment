import logging
from typing import List, Optional
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool

logger = logging.getLogger(__name__)

def estimate_tokens(message: ChatMessage) -> int:
    """
    Estimate the number of tokens in a message.
    Uses a rough approximation of 4 characters per token.
    """
    content = ""
    if isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        for item in message.content:
            if hasattr(item, "text"):
                content += item.text
            elif hasattr(item, "image"):
                # Rough estimate for image tokens
                content += " " * 1000 
    
    return len(content) // 4

def middle_out(messages: List[ChatMessage], max_tokens: Optional[int] = None) -> List[ChatMessage]:
    """
    Truncate messages from the middle to fit within max_tokens.
    Preserves the system message (first message) and the most recent messages.
    
    Args:
        messages: List of chat messages
        max_tokens: Maximum estimated tokens allowed. If None, defaults to 1,000,000.
        
    Returns:
        List of chat messages fitting within the token limit.
    """
    if not messages:
        return []
        
    if max_tokens is None:
        max_tokens = 1_000_000 # Default safe limit for Gemini 1.5 Pro
        
    total_tokens = sum(estimate_tokens(m) for m in messages)
    
    if total_tokens <= max_tokens:
        return messages
        
    logger.info(f"⚠️ Context length {total_tokens} exceeds limit {max_tokens}. Applying middle-out truncation.")
    
    # Always keep the system message if it exists at index 0
    kept_messages = []
    if messages and isinstance(messages[0], ChatMessageSystem):
        kept_messages.append(messages[0])
        messages_to_process = messages[1:]
    else:
        messages_to_process = messages
        
    # Calculate tokens used by kept messages
    current_tokens = sum(estimate_tokens(m) for m in kept_messages)
    
    # Process remaining messages from the end (most recent first)
    # We want to keep the most recent messages
    recent_messages = []
    for msg in reversed(messages_to_process):
        msg_tokens = estimate_tokens(msg)
        if current_tokens + msg_tokens <= max_tokens:
            recent_messages.append(msg)
            current_tokens += msg_tokens
        else:
            # Stop adding messages once we hit the limit
            # We could potentially add a placeholder message here indicating truncation
            break
            
    # Restore original order for recent messages
    recent_messages.reverse()
    
    final_messages = kept_messages + recent_messages
    
    logger.info(f"✂️ Truncated to {len(final_messages)} messages ({current_tokens} tokens). Removed {len(messages) - len(final_messages)} messages.")
    
    return final_messages
