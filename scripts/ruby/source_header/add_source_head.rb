# adds the source head to each source file

root_searching_directory = ARGV[0] || "../../../source"
source_head_file = ARGV[1] || "SOURCE_HEAD.txt"
source_wildcard = ARGV[2] || "/**/*.{cu,cpp,h}"

puts "adding the source head to each source file"
puts "root searching directory: #{root_searching_directory}"
puts "source head file: #{source_head_file}"
puts "source wildcard: #{source_wildcard}"

root_directory = (File.dirname __FILE__) + '/'
puts "root directory: #{root_directory}"

source_head = File.read(root_directory + source_head_file)
puts "source head: "
puts source_head

source_searching_directory = root_directory + root_searching_directory + source_wildcard
puts "source searching directory: #{source_searching_directory}"
source_files = Dir.glob source_searching_directory
#puts source_files
source_files.each do |file|
  puts file
  source = File.read(file)
  if source.start_with? source_head
    return
  end

  source = source_head + source
  File.write file, source
end

puts "added source head, check the git status"
