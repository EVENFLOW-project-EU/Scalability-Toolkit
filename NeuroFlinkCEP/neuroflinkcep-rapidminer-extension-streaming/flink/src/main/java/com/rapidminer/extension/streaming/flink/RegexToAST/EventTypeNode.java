package com.rapidminer.extension.streaming.flink.RegexToAST;

public class EventTypeNode extends Node{
    private String value;

    public EventTypeNode(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
