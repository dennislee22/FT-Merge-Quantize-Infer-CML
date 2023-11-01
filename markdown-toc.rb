#!/usr/bin/env ruby
# encoding utf-8

# allow symlinking to this file
THIS_FILE = (File.symlink?(__FILE__) ? File.readlink(__FILE__) : __FILE__)
THIS_DIR = File.dirname(THIS_FILE)

require File.join(THIS_DIR, 'markdown_toc/arguments_parser.rb')
require File.join(THIS_DIR, 'markdown_toc/tracker.rb')
require File.join(THIS_DIR, 'markdown_toc/node.rb')

require 'optparse'
require 'tempfile'

TOC_INDENT = 4

# NOTE: gitlab won't allow custom html anchors. This version
#       is compatible with the auto-generated ones (and will
#       still work in addition to the github flavored refs).
TOC_ANCHOR_PREFIX = "toc_"

# TOC markers are (ab)using this comment style:
# http://stackoverflow.com/a/20885980/3489294
# These are used to be able to update and remove an existing generated TOC
TOC_START_MARKER = "[//]: # (TOC)"
TOC_END_MARKER = "[//]: # (/TOC)"

def main
  options = MarkdownToc::ArgumentsParser.parse!(ARGV)
  @toc_tracker = MarkdownToc::Tracker.new

  content = File.read(options[:infile])

  if options[:strip] != true
    content = add_toc_data(content, options)
  else
    content = strip_toc_data(content)
  end

  if !options[:outfile].nil?
    write_file(options[:outfile], content)
  else
    puts content
  end
end

def add_toc_data(content, options)
  content = number_chapters(content, options)

  if options[:no_anchors]
    content = strip_chapter_anchors(content)
  end

  write_toc(content, options)
end

def strip_toc_data(content)
  content = strip_chapter_anchors(content)
  strip_toc(content)
end

def number_chapters(content, options)
  node_index = 0
  character_heading_regexp = /
    ^(\#+)\s*                         # heading tag
    (?:<a\sname="#{TOC_ANCHOR_PREFIX}\d+"><\/a>)?  # already generated anchor
    ((?:\d\.?)*)\s*                   # heading number
    (.+)$                             # heading content
  /x

  content.gsub(character_heading_regexp) do
    marker =  Regexp.last_match[1]
    # skip match[2]: this is the "old" numbering
    title = Regexp.last_match[3]

    depth = marker.length # 1-based to account for root node

    new_node = @toc_tracker.add_node(depth, title)
    title = numbered_title(new_node)

    anchor_link = "<a name=\"#{anchor_name(node_index)}\"></a>"
    node_index += 1

    "#{marker} #{anchor_link}#{title}"
  end
end

def strip_chapter_anchors(content)
  content.gsub(/<a\sname="#{Regexp.escape(TOC_ANCHOR_PREFIX)}\d+"><\/a>/, '')
end

def write_toc(content, options)
  nodes = @toc_tracker.get_flat_node_list[1..-1]

  toc = nodes.collect do |node|
    title = numbered_title(node)

    node_index = nodes.index(node)
    link = "##{anchor_name(node_index)}"

    num_indent = (TOC_INDENT * (node.depth - 1))
    indent_char = options[:plain] ? ' ' : '&nbsp;'
    indentation = indent_char * num_indent

    result = "#{indentation}"

    if options[:text_only]
      result += title
    else
      result += "[#{title}](#{link})"
    end

    result
  end

  linebreak_char = options[:plain] ? "\n" : "<br>\n"
  if options[:gitlab_mode]
    linebreak_char += "\n"
  end
  toc = toc.join(linebreak_char)

  if !options[:no_marker]
    toc = [TOC_START_MARKER, toc, TOC_END_MARKER].join("\n")
  end

  content = content.gsub(/^\[TOC\]$/, toc) # replace marker if no toc yet

  # replace generated TOC
  content.gsub(/#{Regexp.escape(TOC_START_MARKER)}.*#{Regexp.escape(TOC_END_MARKER)}/m, toc)
end

def strip_toc(content)
  content.gsub(/#{Regexp.escape(TOC_START_MARKER)}.*#{Regexp.escape(TOC_END_MARKER)}/m, '')
end

def numbered_title(node)
  heading_numbers = node.get_index_path[1..-1]
  heading_numbers.map!{|number| number + 1}
  heading_numbers = heading_numbers.join('.') + "."

  "#{heading_numbers} #{node.content}"
end

def anchor_name(node_index)
  "#{TOC_ANCHOR_PREFIX}#{node_index}"
end

def write_file(file_path, content)
  temp_file = Tempfile.new('markdown-toc')
  temp_file.write(content)
  temp_file.close

  puts "writing #{file_path}..."
  FileUtils.cp(temp_file.path, file_path)
  puts "done."

  temp_file.unlink
end

main
